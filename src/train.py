import argparse
import glob
import os

import lightning as L
import numpy as np
import polars as pl
import toml
import torch
import torchmetrics
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
    "--config", "-c", type=str, help="Path to config file containing parameters."
)
args = parser.parse_args()
config = toml.load(args.config)
batch_size = config["data"]["batch_size"]
LOS_THRESHOLD = config["threshold"]

L.seed_everything(0)


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
    events, static, labels = zip(*batch, strict=False)

    if method == "pad":
        # Function to pad batch-wise due to timeseries of different lengths
        max_events = max([data[0].shape[0] for data in batch])
        n_ftrs = batch[0][0].shape[1]
        events = np.zeros((len(batch), max_events, n_ftrs))
        for i in range(len(batch)):
            j, k = batch[i][0].shape[0], batch[i][0].shape[1]
            events[i] = np.concatenate([batch[i][0], np.zeros((max_events - j, k))])

    elif method == "truncate":
        # Truncate to minimum num of events in batch
        min_events = min([data[0].shape[0] for data in batch])
        events = [event[:min_events] for event in events]

    labels = torch.tensor(labels).unsqueeze(1).to(torch.float32)
    events = torch.tensor(events).to(torch.float32)

    return events, static, labels


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

        return (self.dynamic.to_numpy(), self.static.to_numpy(), self.label)

    def get_all_episodes(self):
        return pl.concat(
            [
                pl.read_csv(f)
                for f in glob.glob(
                    os.path.join(self.root_dir, "*", "episode*_timeseries.csv")
                )
            ]
        )


training_set = MIMICData(args.subjects_root_dir, training_subjects)
training_dataloader = DataLoader(
    training_set, batch_size=batch_size, collate_fn=collate_rugged_timeseries
)


# define the LightningModule
class LitLSTM(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, target_size, lr=0.1):
        super().__init__()
        self.net = LSTM(input_dim, hidden_dim, target_size)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _, y = batch
        x_hat = self.net(x)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


lstm = LitLSTM(input_dim=15, hidden_dim=1024, target_size=1)

# trainer
trainer = L.Trainer(limit_train_batches=100, max_epochs=10)
trainer.fit(model=lstm, train_dataloaders=training_dataloader)
