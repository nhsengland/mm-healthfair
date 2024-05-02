import argparse
import glob
import os

import numpy as np
import polars as pl
import pytorch_lightning as lightning
import torch
import yaml
from models import LSTM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "subjects_root_dir",
    type=str,
    help="Path to the subject-level data",
)
# parser.add_argument("train_subjects",
#                     type=str,
#                     help="Path to file containing list of train subjects.")
parser.add_argument(
    "--config", type=str, help="Path to config file containing parameters."
)
args = parser.parse_args()
config = yaml.load(args.config)
LOS_THRESHOLD = config["threshold"]

lightning.seed_everything(0)


def generate_training_test_subjects(test_ratio=0.15, val_ratio=0.2):
    # Get all subjects with episodes
    all_subjects = set(
        [
            os.path.dirname(i).split("/")[-1]
            for i in glob.glob(
                os.path.join(args.subjects_root_dir, "*", "*timeseries.csv")
            )
        ]
    )
    train_subjects, test_subjects = train_test_split(
        list(all_subjects), test_size=test_ratio
    )
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio)
    return train_subjects, val_subjects, test_subjects


def collate_rugged_timeseries(batch, method="pad"):
    if method == "pad":
        # Function to pad batch-wise due to timeseries of different lengths
        max_events = max([data[0].shape[0] for data in batch])
        n_ftrs = batch[0][0].shape[1]

        _, static, labels = zip(*batch, strict=False)
        labels = torch.tensor(labels).unsqueeze(1)

        events = torch.zeros((len(batch), max_events, n_ftrs))

        for i in range(len(batch)):
            j, k = batch[i][0].shape[0], batch[i][0].shape[1]
            events[i] = torch.cat([batch[i][0], torch.zeros((max_events - j, k))])

    elif method == "truncate":
        # Truncate to minimum num of events in batch
        min_events = min([data[0].shape[0] for data in batch])

        events, static, labels = zip(*batch, strict=False)
        labels = torch.tensor(labels).unsqueeze(1)

        events = torch.tensor([event[:min_events] for event in events])

    return events.float(), static, labels


# Create subject-level training and validation
training_subjects, _, _ = generate_training_test_subjects()


class MIMICData(Dataset):
    def __init__(self, root_dir=None, subjects=None) -> None:
        super().__init__()
        self.subjects = subjects
        self.root_dir = root_dir
        # self.episodes = self.get_all_episodes()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        self.dynamic = pl.read_csv(
            os.path.join(self.root_dir, subject, "episode1_timeseries.csv")
        )

        self.static = pl.read_csv(os.path.join(self.root_dir, subject, "episode1.csv"))
        self.static = self.static.with_columns(
            label=pl.when(pl.col("los") > LOS_THRESHOLD).then(1.0).otherwise(0.0)
        )
        self.label = self.static.select("label").item()
        self.static = self.static.drop(["label", "los"])

        return (
            torch.from_numpy(self.dynamic.to_numpy()),
            torch.from_numpy(self.static.to_numpy()),
            torch.tensor(self.label),
        )

    def get_all_episodes(self):
        return glob.glob(os.path.join(self.root_dir, "*", "episode*_timeseries.csv"))


training_set = MIMICData(args.subjects_root_dir, training_subjects)
training_dataloader = DataLoader(
    training_set, batch_size=4, collate_fn=collate_rugged_timeseries
)


def train(model, trainloader, n_epochs):
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    history = {"loss": []}
    for epoch in range(n_epochs):
        losses = []
        for _, data in enumerate(trainloader, 0):
            inputs, _, labels = data

            model.zero_grad()

            tag_scores = model(inputs)
            loss = loss_function(tag_scores, labels)

            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        avg_loss = np.mean(losses)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1} / {n_epochs}: Loss = {avg_loss:.3f}")
    return history


model = LSTM(input_dim=16, hidden_dim=1024, target_size=1)
train(model, trainloader=training_dataloader, n_epochs=10)
