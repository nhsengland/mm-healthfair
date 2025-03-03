import argparse

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.functions import load_pickle, preview_data


class CollateFn:
    """Custom collate function for static data and labels."""

    def __call__(self, batch):
        static = torch.stack([data[0] for data in batch])
        labels = torch.stack([data[1] for data in batch])

        return static, labels


class CollateTimeSeries:
    """Custom collate function that can handle variable-length timeseries in a batch."""

    def __init__(self, method="pack_pad", min_events=None) -> None:
        self.method = method
        self.min_events = min_events

    def __call__(self, batch):
        static = torch.stack([data[0] for data in batch])
        labels = torch.stack([data[1] for data in batch])
        notes = None
        if len(batch[0]) > 3:  # noqa: PLR2004
            # will also be notes
            notes = torch.stack([data[3] for data in batch])

        # number of dynamic timeseries data (note: dynamic is a list of timeseries)
        n_ts = len(batch[0][2])

        if self.method == "pack_pad":
            dynamic = []
            lengths = []
            for ts in range(n_ts):
                # Function to pad batch-wise due to timeseries of different lengths
                timeseries_lengths = [data[2][ts].shape[0] for data in batch]
                max_events = max(timeseries_lengths)
                n_ftrs = batch[0][2][ts].shape[1]
                events = torch.zeros((len(batch), max_events, n_ftrs))
                for i in range(len(batch)):
                    j, k = batch[i][2][ts].shape[0], batch[i][2][ts].shape[1]
                    events[i] = torch.concat(
                        [batch[i][2][ts], torch.zeros((max_events - j, k))]
                    )
                dynamic.append(events)
                lengths.append(timeseries_lengths)

            if notes is not None:
                return static, labels, dynamic, lengths, notes
            else:
                return static, labels, dynamic, lengths

        elif self.method == "truncate":
            # Truncate to minimum num of events in batch/ specified args

            dynamic = []
            n_ts = len(batch[0][2])
            for ts in range(n_ts):
                min_events = (
                    min([data[2][ts].shape[0] for data in batch])
                    if self.min_events is None
                    else self.min_events
                )
                events = [data[2][ts][:min_events] for data in batch]
                dynamic.append(events)
            return static, labels, dynamic


class MIMIC4Dataset(Dataset):
    """MIMIC-IV Dataset class. Subclass of Pytorch Dataset.
    Reads from .pkl data dictionary where key is hospital admission ID and values are the dataframes.
    """

    def __init__(
        self,
        data_path=None,
        split=None,
        ids=None,
        los_thresh=2,
        static_only=False,
        with_notes=False,
    ) -> None:
        super().__init__()

        self.data_dict = load_pickle(data_path)
        self.hadm_id_list = list(self.data_dict.keys()) if ids is None else ids
        self.dynamic_keys = [
            key
            for key in self.data_dict[int(self.hadm_id_list[0])].keys()
            if "dynamic" in key
        ]
        self.los_thresh = los_thresh
        self.split = split
        self.static_only = static_only
        self.with_notes = with_notes
        self.splits = {"train": None, "val": None, "test": None}

        if ids is None:
            self.setup_data()
            self.splits = {
                "train": self.train_ids,
                "val": self.val_ids,
                "test": self.test_ids,
            }
        else:
            self.splits[split] = ids

    def __len__(self):
        return (
            len(self.splits[self.split])
            if self.split is not None
            else len(self.hadm_id_list)
        )

    def __getitem__(self, idx):
        hadm_id = int(self.splits[self.split][idx])

        static = self.data_dict[hadm_id]["static"]  # polars df
        static = static.with_columns(
            label=pl.when(pl.col("los") > self.los_thresh).then(1.0).otherwise(0.0)
        )

        label = torch.tensor(
            static.select("label").item(), dtype=torch.float32
        ).unsqueeze(-1)

        static = static.drop(["label", "los"])
        static = torch.tensor(static.to_numpy(), dtype=torch.float32)

        if self.static_only:
            return static, label

        else:
            dynamic = [
                self.data_dict[hadm_id][i] for i in self.dynamic_keys
            ]  # list of polars df's
            dynamic = [torch.tensor(x.to_numpy(), dtype=torch.float32) for x in dynamic]

            if self.with_notes:
                notes = self.data_dict[hadm_id]["notes"]  # 1 x 768
                notes = torch.tensor(notes, dtype=torch.float32)
                return static, label, dynamic, notes
            else:
                return static, label, dynamic

    def print_label_dist(self):
        # if no particular split then use entire data dict
        if self.split is None:
            id_list = self.hadm_id_list
        else:
            id_list = self.splits[self.split]

        all_static = pl.concat(
            [self.data_dict[int(i)]["static"].select(pl.col("los")) for i in id_list]
        )
        n_positive = all_static.select(pl.col("los") > self.los_thresh).sum().item()

        if self.split is not None:
            print(f"{self.split.upper()}:")

        print(f"Positive cases (los > {self.los_thresh}): {n_positive}")
        print(
            f"Negative cases (los <= {self.los_thresh}): {all_static.height - n_positive}"
        )

    def get_feature_dim(self, key="static"):
        return self.data_dict[int(self.hadm_id_list[0])][key].shape[1]

    def get_feature_list(self, key="static"):
        data = self.data_dict[int(self.hadm_id_list[0])][key]
        if key == "static":
            data = data.drop("los")
        return data.columns

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
