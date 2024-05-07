import argparse
import glob
import os

import lightning as L
import numpy as np
import polars as pl
import toml
import torch
import torchmetrics
import utils
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
n_epochs = config["train"]["epochs"]
lr = config["train"]["learning_rate"]
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
training_subjects, val_subjects, _ = generate_training_test_subjects()


class MIMICData(Dataset):
    def __init__(self, root_dir=None, subjects=None) -> None:
        super().__init__()
        self.subjects = subjects
        self.root_dir = root_dir
        self.dynamic_data = []
        self.static_data = []

    def __len__(self):
        return len(self.dynamic_data)

    def __getitem__(self, idx):
        self.dynamic = self.dynamic_data[idx].collect()
        self.static = self.static_data[idx].collect()

        self.static = self.static.with_columns(
            label=pl.when(pl.col("los") > LOS_THRESHOLD).then(1.0).otherwise(0.0)
        )
        self.label = self.static.select("label").item()
        self.static = self.static.drop(["label", "los"])

        return (self.dynamic.to_numpy(), self.static.to_numpy(), self.label)

    def prepare_data(self, preprocess=True):
        filepaths = [
            f
            for f in glob.glob(
                os.path.join(self.root_dir, "*", "episode1_timeseries.csv")
            )
            if os.path.dirname(f).split("/")[-1] in self.subjects
        ]

        self.static_data = [
            pl.scan_csv(
                os.path.join(
                    os.path.dirname(f), os.path.basename(f).split("_")[0] + ".csv"
                )
            )
            for f in filepaths
        ]

        # Preprocess the data
        if preprocess:
            self.dynamic_data = [
                self.preprocess_data(
                    pl.scan_csv(f).with_columns(pl.all().cast(pl.Float64))
                )
                for f in filepaths
            ]

        else:
            self.dynamic_data = [
                pl.scan_csv(f).with_columns(pl.all().cast(pl.Float64))
                for f in filepaths
            ]

    def preprocess_data(self, input_data, impute_strategy="ffill"):
        # assumes data is a lazyframe

        # TODO: Ensure this works for lazyframe
        # Aggregate into time-windows e.g., every 2hr by upsampling then downsampling
        # input_data = input_data.collect().upsample(time_column="charttime", every='2h').lazy()
        # input_data = (input_data.group_by_dynamic(
        #             "charttime",
        #             every="2h"
        #         ).agg(pl.exclude('charttime')).mean())

        # Imputating of missing values using masking (adds features) or filling
        if impute_strategy is not None:
            if impute_strategy == "mask":
                # Add new feature column with mask for whether row is nan or not
                for i in input_data.columns:
                    input_data = input_data.with_columns(
                        pl.col(i).is_null().alias(i + "_isna")
                    )

            elif impute_strategy == "ffill":
                # Fill missing values using forward fill
                input_data = input_data.fill_null(strategy="forward")
                input_data = input_data.fill_null(strategy="backward")

                # for remaining null values use -999
                input_data = input_data.fill_null(value=-999)

            elif impute_strategy == "mean":
                mean_map = utils.get_pop_means(self.subjects_root_path)
                input_data = input_data.fill_null(
                    mean_map
                )  # TODO: check this is working

            else:
                raise ValueError(
                    "impute_strategy must be one of [None, mask, ffill, mean]"
                )

        return input_data


training_set = MIMICData(args.subjects_root_dir, training_subjects)
training_set.prepare_data()
training_dataloader = DataLoader(
    training_set, batch_size=batch_size, collate_fn=collate_rugged_timeseries
)

validation_set = MIMICData(args.subjects_root_dir, val_subjects)
validation_set.prepare_data()
val_dataloader = DataLoader(
    validation_set, batch_size=batch_size, collate_fn=collate_rugged_timeseries
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

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x_hat = self.net(x)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


lstm = LitLSTM(input_dim=15, hidden_dim=256, target_size=1, lr=lr)

# trainer
trainer = L.Trainer(limit_train_batches=100, max_epochs=n_epochs)
trainer.fit(
    model=lstm, train_dataloaders=training_dataloader, val_dataloaders=val_dataloader
)
