import argparse
import pickle

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.functions import preview_data


class CollateTimeSeries:
    def __init__(self, method="pack_pad", min_events=None) -> None:
        self.method = method
        self.min_events = min_events

    def __call__(self, batch):
        static = torch.stack([data[0] for data in batch])
        labels = torch.stack([data[2] for data in batch])

        if self.method == "pack_pad":
            # Function to pad batch-wise due to timeseries of different lengths
            timeseries_lengths = [data[1].shape[0] for data in batch]
            max_events = max(timeseries_lengths)
            n_ftrs = batch[0][1].shape[1]
            events = torch.zeros((len(batch), max_events, n_ftrs))
            for i in range(len(batch)):
                j, k = batch[i][1].shape[0], batch[i][1].shape[1]
                events[i] = torch.concat(
                    [batch[i][1], torch.zeros((max_events - j, k))]
                )
            return static, events, timeseries_lengths, labels

        elif self.method == "truncate":
            # Truncate to minimum num of events in batch/ specified args
            min_events = (
                min([data[0].shape[0] for data in batch])
                if self.min_events is None
                else self.min_events
            )
            events = [event[:min_events] for event in events]
            return static, events, labels


class MIMIC4Dataset(Dataset):
    def __init__(self, split=None, ids=None, data_path=None, los_thresh=2) -> None:
        super().__init__()

        with open(data_path, "rb") as f:
            self.data_dict = pickle.load(f)

        self.hadm_id_list = list(self.data_dict.keys()) if ids is None else ids
        self.los_thresh = los_thresh
        self.split = split
        if ids is None:
            self.setup_data()
            self.splits = {
                "train": self.train_ids,
                "val": self.val_ids,
                "test": self.test_ids,
            }
        else:
            self.splits = {split: ids}

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
        self.label = torch.tensor(
            self.static.select("label").item(), dtype=torch.float32
        ).unsqueeze(-1)

        self.static = self.static.drop(["label", "los"])
        self.static = torch.tensor(self.static.to_numpy(), dtype=torch.float32)
        self.dynamic = torch.tensor(self.dynamic.to_numpy(), dtype=torch.float32)

        return self.static, self.dynamic, self.label

    def get_split_ids(self, split):
        return self.splits[split]

    def setup_data(self, test_ratio=0.15, val_ratio=0.2):
        train_ids, test_ids = train_test_split(self.hadm_id_list, test_size=test_ratio)
        train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids


if __name__ == "__main__":
    # Preview data from a saved pkl file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "processed_data_path",
        type=str,
        help="Path to the pickled data.",
    )
    args = parser.parse_args()
    preview_data(args.processed_data_path)
