import argparse
import pickle

import lightning as L
import numpy as np
import polars as pl
import toml
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from models import LitLSTM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def collate_rugged_timeseries(batch, method="pack_pad"):
    events, static, labels = zip(*batch, strict=False)

    if (method == "pad_only") | (method == "pack_pad"):
        # Function to pad batch-wise due to timeseries of different lengths
        event_lengths = [data[0].shape[0] for data in batch]
        max_events = max(event_lengths)
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

    if method == "pack_pad":
        # now use torch.nn.utils.rnn.pack_padded_sequence() to pack according to the length
        events = torch.nn.utils.rnn.pack_padded_sequence(
            events, event_lengths, batch_first=True, enforce_sorted=False
        )

    return events, static, labels


class MIMICData(Dataset):
    def __init__(self, split=None, data_path=None, los_thresh=2) -> None:
        super().__init__()

        with open(data_path, "rb") as f:
            self.data_dict = pickle.load(f)

        self.hadm_id_list = list(self.data_dict.keys())
        self.los_thresh = los_thresh
        self.split = split
        if self.split is not None:
            self.setup_data()
            self.splits = {
                "train": self.train_ids,
                "val": self.val_ids,
                "test": self.test_ids,
            }

    def __len__(self):
        return (
            len(self.splits[self.split])
            if self.split is not None
            else len(self.hadm_id_list)
        )

    def __getitem__(self, idx):
        hadm_id = self.splits[self.split][idx]
        self.dynamic = self.data_dict[hadm_id]["dynamic"]  # polars df
        self.static = self.data_dict[hadm_id]["static"]  # polars df

        self.static = self.static.with_columns(
            label=pl.when(pl.col("los") > self.los_thresh).then(1.0).otherwise(0.0)
        )
        self.label = self.static.select("label").item()
        self.static = self.static.drop(["label", "los"])

        return (self.dynamic.to_numpy(), self.static.to_numpy(), self.label)

    def setup_data(self, test_ratio=0.15, val_ratio=0.2):
        train_ids, test_ids = train_test_split(self.hadm_id_list, test_size=test_ratio)
        train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subjects_root_dir",
        type=str,
        help="Path to the subject-level data",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.toml",
        help="Path to config toml file containing parameters.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Whether to use cpu. Defaults to gpu"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb for logging. Defaults to False",
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    device = "gpu" if not args.cpu else "cpu"
    use_wandb = args.wandb

    batch_size = config["data"]["batch_size"]
    n_epochs = config["train"]["epochs"]
    lr = config["train"]["learning_rate"]
    num_workers = config["data"]["num_workers"]
    los_threshold = config["threshold"]
    exp_name = config["train"]["experiment_name"]

    L.seed_everything(0)

    # Create subject-level training and validation

    training_set = MIMICData("train", args.subjects_root_dir, los_thresh=los_threshold)
    training_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_rugged_timeseries,
    )

    validation_set = MIMICData("val", args.subjects_root_dir, los_thresh=los_threshold)
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_rugged_timeseries,
    )

    lstm = LitLSTM(input_dim=16, hidden_dim=256, target_size=1, lr=lr)

    # trainer
    if use_wandb:
        logger = WandbLogger(
            log_model=True, project="nhs-mm-healthfair", save_dir="logs"
        )
    else:
        logger = CSVLogger("logs")

    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=n_epochs,
        log_every_n_steps=10,
        logger=logger,
        accelerator=device,
    )
    trainer.fit(
        model=lstm,
        train_dataloaders=training_dataloader,
        val_dataloaders=val_dataloader,
    )
