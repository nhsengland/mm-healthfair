# Class definitions for datatypes
# Code adapted from: https://github.com/Wang-Yuanlong/MultimodalPred/blob/master/dataset/ehr_dataset.py

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dask import dataframe as dd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MM_dataset(Dataset):
    def __init__(
        self,
        split="train",
        img_transform=None,
        task="mortality",
        longstay_mintime=0,
        ehr_data_path=None,
    ):
        super().__init__()
        self.ehr = EHR_dataset(
            split="all",
            task=task,
            longstay_mintime=longstay_mintime,
            data_path=ehr_data_path,
        )
        # self.cxr = CXR_dataset(split='all', transform=img_transform, task=task)
        # self.note = Note_dataset(split='all', task=task)
        datatable_ehr = self.ehr.get_samplelist()[["subject_id", "hadm_id"]]
        # datatable_cxr = self.cxr.get_samplelist()[['subject_id', 'hadm_id', 'label']]
        # datatable_note = self.note.get_samplelist()[['subject_id', 'hadm_id']]
        # self.data_table = pd.merge(datatable_cxr, datatable_ehr, how='inner', on=['subject_id', 'hadm_id'])
        # self.data_table = pd.merge(self.data_table, datatable_note, how='inner', on=['subject_id', 'hadm_id'])
        self.data_table = datatable_ehr

        # self.label = pd.merge(self.data_table, self.ehr.all_adm_disch, how='left', on=['subject_id', 'hadm_id'])
        # self.label = self.label.apply(lambda x: (1 if x['discharge_location']=='DIED' else 0) if not x.empty else None, axis=1)
        # self.label = self.data_table.pop('label')
        self.label = self.ehr.get_label()

        if split != "all":
            X_train, X_test, Y_train, Y_test = train_test_split(
                np.arange(len(self.label)).reshape(-1, 1),
                self.label,
                test_size=0.15,
                random_state=0,
            )
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.15, random_state=0
            )
            if split == "train":
                self.data_table, self.label = (
                    self.data_table.iloc[X_train.reshape(-1)],
                    Y_train,
                )
            elif split == "test":
                self.data_table, self.label = (
                    self.data_table.iloc[X_test.reshape(-1)],
                    Y_test,
                )
            if split == "val":
                self.data_table, self.label = (
                    self.data_table.iloc[X_val.reshape(-1)],
                    Y_val,
                )
        else:
            self.label = self.label.to_numpy()
        pass

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        x = self.data_table.iloc[index]
        subject_id = x.subject_id
        hadm_id = x.hadm_id

        demo, ce_ts, le_ts, pe_ts, timestamps = self.ehr.query(subject_id, hadm_id)
        # img_list, img_position, img_time = self.cxr.query(subject_id, hadm_id)
        # notes = self.note.query(subject_id, hadm_id)

        # target = self.ehr.all_adm_disch[(self.ehr.all_adm_disch['subject_id'] == subject_id) &
        #                                 (self.ehr.all_adm_disch['hadm_id'] == hadm_id)]
        # target = 1 if target['discharge_location'].iloc[0] == 'DIED' else 0

        target = self.label[index]
        return (demo, ce_ts, le_ts, pe_ts, timestamps), None, None, target

    def get_collate(self):
        def ehr_collate(samples):
            demo, ce_ts, le_ts, pe_ts, timestamps = map(list, zip(*map(list, samples)))
            return torch.from_numpy(np.stack(demo)), ce_ts, le_ts, pe_ts, timestamps

        def cxr_collate(samples):
            img_list, img_positions, img_times = map(list, zip(*map(list, samples)))
            return img_list, img_positions, img_times

        def note_collate(samples):
            return samples

        def collate_fn(samples):
            ehr_samples, _, _, targets = map(list, zip(*map(list, samples)))
            return ehr_collate(ehr_samples), None, None, torch.LongTensor(targets)

            # ehr_samples, cxr_samples, note_samples, targets = map(list, zip(*map(list, samples)))
            # return ehr_collate(ehr_samples), cxr_collate(cxr_samples), note_collate(note_samples), torch.LongTensor(targets)

        return collate_fn


# if __name__ == '__main__':
#     import torchvision.transforms as transforms
#     img_transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize(norm_mean, norm_std),
#     ])
#     x = MUL_dataset(img_transform=img_transform, task='readmission')
#     for i in range(len(x)):
#         a = x[i]


def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1 - t0).total_seconds() / 3600  # Result in hrs
    except Exception:
        delta_t = np.nan

    return delta_t


def generate_categorial_num(table, fields):
    for field in fields:
        if type(field) == tuple:
            value_list = list(map(lambda x: tuple(x), list(table[list(field)].values)))
            cat = pd.Categorical(value_list)
            table["_".join(field) + "_num"] = cat.codes
        else:
            cat = pd.Categorical(table[field].values)
            table[field + "_num"] = cat.codes


def generate_age_num(table):
    table["anchor_age_num"] = table.apply(
        lambda x: int(x["anchor_age"] / 10) if not x.empty else None, axis=1
    )


def value_filter(table, by, subset, normal_values):
    # table['value'] = table.apply(lambda x: (np.nan if x['label'] in normal_values.keys() and
    #                                                   (float(x['value']) > normal_values[x['label']][1] or
    #                                                   float(x['value']) < normal_values[x['label']][0])
    #                                                else x['value'])
    #                                        if not x.empty
    #                                        else None, axis=1)
    def filter(x):
        idx = x.index
        y = table.loc[idx][by]
        by_value = tuple(*(y.iloc[[0]].values))
        if len(by_value) == 1:
            by_value = by_value[0]
        if by_value in normal_values.keys():
            x = x.astype("float64")
            x[(x < normal_values[by_value][0]) | (x > normal_values[by_value][1])] = (
                np.nan
            )
        x = x.astype("str")
        return x

    out = table.groupby(by=by, as_index=False)[subset].transform(filter)
    return out


def discretize(table, subset, by, split_num=200):
    have_diff_types = "param_type" in table.columns

    def discretize_func(x):
        if have_diff_types:
            x_type = table.loc[x.index].iloc[0]["param_type"]
            if x_type == "Text":
                return pd.Categorical(x).codes
        x = x.astype("float64")
        x_min = x.min()
        x_max = x.max()
        if x_min == x_max:
            y = np.zeros_like(x.values)
            nan_mask = x.isna()
            y[nan_mask] = np.nan
            return y
        y = (x - x.min()) / (x.max() - x.min())
        return pd.cut(y, bins=split_num, labels=False)

    out = table.groupby(by=by, as_index=False)[subset].transform(discretize_func)
    return out


def replace_str_with_na(table: pd.DataFrame, subset, by, encoding=["___"]):  # noqa: B006
    def na_func(x):
        for code in encoding:
            x.replace(code, np.nan, inplace=True)
        return x

    out = table.groupby(by=by, as_index=False)[subset].transform(na_func)
    return out


def fill_na(table: pd.DataFrame, subset, by):
    def fill_func(x):
        na_num = x.isna().sum()
        if na_num < 1:
            return x.astype("int64")
        if na_num == len(x):
            x_label = table.loc[x.index].iloc[0]["label"]
            y = round(table[table["label"] == x_label][subset].mean().iloc[0])
            return x.fillna(y).astype("int64")
        return x.fillna(round(x.mean())).astype("int64")

    out = table.groupby(by=by, as_index=False)[subset].transform(fill_func)
    return out


class EHR_dataset(Dataset):
    def __init__(  # noqa: PLR0912, PLR0915
        self,
        split="train",
        data_path=None,
        regenerate=False,
        reprocess=False,
        task="mortality",
        longstay_mintime=0,
    ):  # noqa: PLR0915, PLR0912
        super().__init__()
        self.data_path = data_path

        self.clipped_dir = f"{data_path}/tmp/clipped"
        self.processed_dir = f"{data_path}/tmp/processed"
        #  If data dirs do not exist, create it
        if not os.path.exists(self.clipped_dir):
            print(f"Creating directory at {self.clipped_dir}...")
            Path(self.clipped_dir).mkdir(parents=True)
        if not os.path.exists(self.processed_dir):
            print(f"Creating directory at {self.processed_dir}...")
            Path(self.processed_dir).mkdir(parents=True)

        self.max_time_delta = 48  # hours

        self.key_ids = dd.read_csv(
            os.path.join(data_path, "haim_mimiciv_key_ids.csv")
        ).astype("int64")
        demographic = dd.read_csv(
            os.path.join(data_path, "core.csv"), dtype={"deathtime": "object"}
        )
        with open("src/benchmark/normal_range.json") as f:
            self.normal_values = json.load(f)

        self.procedure_event_list = [
            "Foley Catheter",
            "PICC Line",
            "Intubation",
            "Peritoneal Dialysis",
            "Bronchoscopy",
            "EEG",
            "Dialysis - CRRT",
            "Dialysis Catheter",
            "Chest Tube Removed",
            "Hemodialysis",
        ]
        self.lab_event_list = [
            "Glucose",
            "Potassium",
            "Sodium",
            "Chloride",
            "Creatinine",
            "Urea Nitrogen",
            "Bicarbonate",
            "Anion Gap",
            "Hemoglobin",
            "Hematocrit",
            "Magnesium",
            "Platelet Count",
            "Phosphate",
            "White Blood Cells",
            "Calcium, Total",
            "MCH",
            "Red Blood Cells",
            "MCHC",
            "MCV",
            "RDW",
            "Platelet Count",
            "Neutrophils",
            "Vancomycin",
        ]
        self.chart_event_list = [
            "Heart Rate",
            "Non Invasive Blood Pressure systolic",
            "Non Invasive Blood Pressure diastolic",
            "Non Invasive Blood Pressure mean",
            "Respiratory Rate",
            "O2 saturation pulseoxymetry",
            "GCS - Verbal Response",
            "GCS - Eye Opening",
            "GCS - Motor Response",
        ]

        if regenerate & reprocess:
            print(f"Regenerating clipped dataset, saving to {self.clipped_dir} ...")
            self.regenerate_data()

        chartevents = dd.read_csv(
            os.path.join(self.clipped_dir, "chartevents.csv"),
            assume_missing=True,
            low_memory=False,
            dtype={"value": "object", "valueuom": "object"},
        )
        labevents = dd.read_csv(
            os.path.join(self.clipped_dir, "labevents.csv"),
            assume_missing=True,
            dtype={
                "storetime": "object",
                "value": "object",
                "valueuom": "object",
                "flag": "object",
                "priority": "object",
                "comments": "object",
            },
        )
        procedureevents = dd.read_csv(
            os.path.join(self.clipped_dir, "procedureevents.csv"),
            assume_missing=True,
            dtype={
                "value": "object",
                "secondaryordercategoryname": "object",
                "totalamountuom": "object",
            },
        )

        if reprocess:
            print(f"Reprocessing clipped dataset, saving to {self.processed_dir} ...")
            self.reprocess_data(demographic, chartevents, labevents, procedureevents)
        else:
            self.demographic = pd.read_csv(
                os.path.join(self.processed_dir, "demographic.csv")
            )
            self.chartevents = pd.read_csv(
                os.path.join(self.processed_dir, "chartevents.csv"),
                dtype={"value": "object", "valueuom": "object"},
            )
            self.labevents = pd.read_csv(
                os.path.join(self.processed_dir, "labevents.csv"),
                dtype={
                    "storetime": "object",
                    "value": "object",
                    "valueuom": "object",
                    "flag": "object",
                    "priority": "object",
                    "comments": "object",
                },
            )
            self.procedureevents = pd.read_csv(
                os.path.join(self.processed_dir, "procedureevents.csv"),
                dtype={
                    "value": "object",
                    "secondaryordercategoryname": "object",
                    "totalamountuom": "object",
                },
            )
            self.all_adm_disch = pd.read_csv(
                os.path.join(self.processed_dir, "all_adm_disch.csv")
            )

            # self.labevents = self.labevents[self.labevents['fluid']=='Blood']
            # self.labevents = self.labevents.dropna(axis=0, subset=['value'])

            self.chart_event_list = self.chart_event_list
            # self.lab_event_list = self.labevents[['label', 'fluid']].drop_duplicates().reset_index(drop=True)
            # self.lab_event_list = list(map(lambda x: tuple(x), list(self.lab_event_list.values)))
            self.lab_event_list = self.lab_event_list
            self.procedure_event_list = self.procedure_event_list

        # generate fround truth label here
        # all_disch = self.demographic[['subject_id', 'hadm_id', 'dischtime', 'discharge_location']].drop_duplicates().reset_index(drop=True)
        # all_timestamps = self.all_timestamps.merge(all_disch, how='left')
        # all_timestamps['delta_time'] = all_timestamps.apply(lambda x: date_diff_hrs(x['dischtime'], x['time']) if not x.empty else None, axis=1)
        # all_timestamps['target'] = all_timestamps.apply(lambda x: (1 if (x['delta_time'] < 48) & (x['discharge_location'] == 'DIED') else 0) if not x.empty else None, axis=1)
        # self.all_timestamps = all_timestamps
        if task == "mortality":
            self.all_adm_disch["label"] = self.all_adm_disch.apply(
                lambda x: (1 if x["discharge_location"] == "DIED" else 0)
                if not x.empty
                else None,
                axis=1,
            )
        elif task == "longstay":
            self.all_adm_disch["admittime"] = pd.to_datetime(
                self.all_adm_disch["admittime"]
            )
            self.all_adm_disch["dischtime"] = pd.to_datetime(
                self.all_adm_disch["dischtime"]
            )
            self.all_adm_disch["time_stay"] = self.all_adm_disch.apply(
                lambda x: date_diff_hrs(x["dischtime"], x["admittime"])
                if not x.empty
                else None,
                axis=1,
            )
            self.all_adm_disch = self.all_adm_disch[
                self.all_adm_disch["time_stay"] >= longstay_mintime
            ]
            self.all_adm_disch["label"] = self.all_adm_disch.apply(
                lambda x: (1 if x["time_stay"] > 7 * 24 else 0)
                if not x.empty
                else None,
                axis=1,
            ).astype("int64")
        elif task == "readmission":

            def readmission_transform(x: pd.DataFrame):
                out = x.sort_values("admittime")
                out["nexttime"] = out["admittime"].shift(-1)
                out["label"] = out.apply(
                    lambda x: 0
                    if pd.isna(x["nexttime"])
                    else (
                        1
                        if date_diff_hrs(x["nexttime"], x["dischtime"]) <= 30 * 24
                        else 0
                    ),
                    axis=1,
                ).astype("int64")
                return out["label"]

            self.all_adm_disch["admittime"] = pd.to_datetime(
                self.all_adm_disch["admittime"]
            )
            self.all_adm_disch["dischtime"] = pd.to_datetime(
                self.all_adm_disch["dischtime"]
            )
            self.all_adm_disch["label"] = self.all_adm_disch.groupby(
                ["subject_id"], as_index=False, group_keys=False
            ).apply(readmission_transform)

        self.label = self.all_adm_disch.pop("label")

        if split == "all":
            self.data_table, self.label = self.all_adm_disch, self.label.to_numpy()
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                np.arange(len(self.label)).reshape(-1, 1),
                self.label.to_numpy(),
                test_size=0.15,
                random_state=0,
            )
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.15, random_state=0
            )
            if split == "train":
                self.data_table, self.label = (
                    self.all_adm_disch.iloc[X_train.reshape(-1)],
                    Y_train,
                )
            elif split == "test":
                self.data_table, self.label = (
                    self.all_adm_disch.iloc[X_test.reshape(-1)],
                    Y_test,
                )
            if split == "val":
                self.data_table, self.label = (
                    self.all_adm_disch.iloc[X_val.reshape(-1)],
                    Y_val,
                )

        self.data_table.reset_index()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = self.data_table.iloc[index]
        subject_id = sample.subject_id
        hadm_id = sample.hadm_id
        target = self.label[index]

        demo = self.demographic[
            (self.demographic["subject_id"] == subject_id)
            & (self.demographic["hadm_id"] == hadm_id)
        ]

        ce = self.chartevents[
            (self.chartevents["subject_id"] == subject_id)
            & (self.chartevents["hadm_id"] == hadm_id)
        ]

        le = self.labevents[
            (self.labevents["subject_id"] == subject_id)
            & (self.labevents["hadm_id"] == hadm_id)
        ]

        pe = self.procedureevents[
            (self.procedureevents["subject_id"] == subject_id)
            & (self.procedureevents["hadm_id"] == hadm_id)
        ]

        demo = demo[
            [
                "anchor_age_num",
                "insurance_num",
                "language_num",
                "marital_status_num",
                "race_num",
            ]
        ]
        ce = ce[["label_num", "value_num", "delta_time"]]
        le = le[["label_num", "value_num", "delta_time"]]
        pe = pe[["label_num", "delta_starttime", "delta_endtime"]]

        timestamps = np.unique(
            np.concatenate(
                [
                    ce["delta_time"].values,
                    le["delta_time"].values,
                    pe["delta_starttime"].values,
                ]
            )
        )

        ce_ts, le_ts, pe_ts = [], [], []
        for time in timestamps:
            ce_time = ce[ce["delta_time"] == time]
            le_time = le[le["delta_time"] == time]

            ce_pack = []
            for row in ce_time.itertuples():
                ce_pack.append((row.label_num, row.value_num))
            ce_ts.append(ce_pack)

            le_pack = []
            for row in le_time.itertuples():
                le_pack.append((row.label_num, row.value_num))
            le_ts.append(le_pack)

            pe_pack = []
            for row in pe.itertuples():
                if time >= row.delta_starttime and time < row.delta_endtime:
                    pe_pack.append(row.label_num)
            pe_ts.append(pe_pack)

        demo = demo.values[0]
        return demo, ce_ts, le_ts, pe_ts, timestamps, target

    def query(self, subject_id, hadm_id):
        demo = self.demographic[
            (self.demographic["subject_id"] == subject_id)
            & (self.demographic["hadm_id"] == hadm_id)
        ]

        ce = self.chartevents[
            (self.chartevents["subject_id"] == subject_id)
            & (self.chartevents["hadm_id"] == hadm_id)
        ]
        #   (self.chartevents['charttime'] > admittime) &\
        #   (self.chartevents['charttime'] < dischtime) &\
        #   (self.chartevents['charttime'] <= time)]

        le = self.labevents[
            (self.labevents["subject_id"] == subject_id)
            & (self.labevents["hadm_id"] == hadm_id)
        ]
        # (self.labevents['charttime'] > admittime) &\
        # (self.labevents['charttime'] < dischtime) &\
        # (self.labevents['charttime'] <= time)]

        pe = self.procedureevents[
            (self.procedureevents["subject_id"] == subject_id)
            & (self.procedureevents["hadm_id"] == hadm_id)
        ]
        # (self.procedureevents['starttime'] > admittime) &\
        # (self.procedureevents['endtime'] < dischtime) &\
        # (self.procedureevents['starttime'] <= time) &\
        # (self.procedureevents['endtime'] >= time)]

        # ce['charttime'] = ce.apply(lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
        # le['charttime'] = le.apply(lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
        # pe['starttime'] = pe.apply(lambda x: date_diff_hrs(x['starttime'], admittime) if not x.empty else None, axis=1)
        # pe['endtime'] = pe.apply(lambda x: date_diff_hrs(x['endtime'], admittime) if not x.empty else None, axis=1)

        demo = demo[
            [
                "anchor_age_num",
                "insurance_num",
                "language_num",
                "marital_status_num",
                "race_num",
            ]
        ]
        ce = ce[["label_num", "value_num", "delta_time"]]
        le = le[["label_num", "value_num", "delta_time"]]
        pe = pe[["label_num", "delta_starttime", "delta_endtime"]]

        timestamps = np.unique(
            np.concatenate(
                [
                    ce["delta_time"].values,
                    le["delta_time"].values,
                    pe["delta_starttime"].values,
                ]
            )
        )

        ce_ts, le_ts, pe_ts = [], [], []
        for time in timestamps:
            ce_time = ce[ce["delta_time"] == time]
            le_time = le[le["delta_time"] == time]

            ce_pack = []
            for row in ce_time.itertuples():
                ce_pack.append((row.label_num, row.value_num))
            ce_ts.append(ce_pack)

            le_pack = []
            for row in le_time.itertuples():
                le_pack.append((row.label_num, row.value_num))
            le_ts.append(le_pack)

            pe_pack = []
            for row in pe.itertuples():
                if time >= row.delta_starttime and time < row.delta_endtime:
                    pe_pack.append(row.label_num)
            pe_ts.append(pe_pack)

        demo = demo.values[0]

        return demo, ce_ts, le_ts, pe_ts, timestamps

    def get_samplelist(self):
        return self.data_table

    def get_label(self):
        return self.label

    def get_collate(self):
        def collate_fn(samples):
            demo, ce_ts, le_ts, pe_ts, timestamps, target = map(
                list, zip(*map(list, samples))
            )
            return (
                torch.from_numpy(np.stack(demo)),
                ce_ts,
                le_ts,
                pe_ts,
                timestamps,
                torch.LongTensor(target),
            )

        return collate_fn

    def reprocess_data(  # noqa: PLR0915
        self, demographic=None, chartevents=None, labevents=None, procedureevents=None
    ):  # noqa: PLR0915
        all_admissions = (
            self.key_ids[["subject_id", "hadm_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        demographic["admittime"] = dd.to_datetime(demographic["admittime"])
        demographic["dischtime"] = dd.to_datetime(demographic["dischtime"])
        chartevents["charttime"] = dd.to_datetime(chartevents["charttime"])
        labevents["charttime"] = dd.to_datetime(labevents["charttime"])
        procedureevents["starttime"] = dd.to_datetime(procedureevents["starttime"])
        procedureevents["endtime"] = dd.to_datetime(procedureevents["endtime"])
        demographic[["subject_id", "hadm_id"]] = demographic[
            ["subject_id", "hadm_id"]
        ].astype("int64")
        chartevents[["subject_id", "hadm_id"]] = chartevents[
            ["subject_id", "hadm_id"]
        ].astype("int64")
        labevents[["subject_id", "hadm_id"]] = labevents[
            ["subject_id", "hadm_id"]
        ].astype("int64")
        procedureevents[["subject_id", "hadm_id"]] = procedureevents[
            ["subject_id", "hadm_id"]
        ].astype("int64")

        demographic = (
            all_admissions.merge(demographic, on=["subject_id", "hadm_id"], how="left")
            .drop_duplicates()
            .reset_index(drop=True)
        )
        chartevents = all_admissions.merge(
            chartevents, on=["subject_id", "hadm_id"], how="left"
        )
        labevents = all_admissions.merge(
            labevents, on=["subject_id", "hadm_id"], how="left"
        )
        procedureevents = all_admissions.merge(
            procedureevents, on=["subject_id", "hadm_id"], how="left"
        )

        all_adm_disch = (
            demographic[
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "dischtime",
                    "discharge_location",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        all_admittime = all_adm_disch[["subject_id", "hadm_id", "admittime"]]

        chartevents = all_admittime.merge(
            chartevents, on=["subject_id", "hadm_id"], how="left"
        )
        labevents = all_admittime.merge(
            labevents, on=["subject_id", "hadm_id"], how="left"
        )
        procedureevents = all_admittime.merge(
            procedureevents, on=["subject_id", "hadm_id"], how="left"
        )

        chartevents["delta_time"] = chartevents.apply(
            lambda x: date_diff_hrs(x["charttime"], x["admittime"])
            if not x.empty
            else None,
            axis=1,
            meta=(None, "float64"),
        )
        labevents["delta_time"] = labevents.apply(
            lambda x: date_diff_hrs(x["charttime"], x["admittime"])
            if not x.empty
            else None,
            axis=1,
            meta=(None, "float64"),
        )
        procedureevents["delta_starttime"] = procedureevents.apply(
            lambda x: date_diff_hrs(x["starttime"], x["admittime"])
            if not x.empty
            else None,
            axis=1,
            meta=(None, "float64"),
        )
        procedureevents["delta_endtime"] = procedureevents.apply(
            lambda x: date_diff_hrs(x["endtime"], x["admittime"])
            if not x.empty
            else None,
            axis=1,
            meta=(None, "float64"),
        )

        chartevents = chartevents[
            (chartevents["delta_time"] <= self.max_time_delta)
            & (chartevents["delta_time"] > 0)
        ]
        labevents = labevents[
            (labevents["delta_time"] <= self.max_time_delta)
            & (labevents["delta_time"] > 0)
        ]
        procedureevents = procedureevents[
            (procedureevents["delta_starttime"] <= self.max_time_delta)
            & (procedureevents["delta_starttime"] > 0)
        ]

        demographic = (
            demographic[
                [
                    "subject_id",
                    "hadm_id",
                    "anchor_age",
                    "insurance",
                    "language",
                    "marital_status",
                    "race",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # all_timestamps = pd.DataFrame(columns=['subject_id', 'hadm_id', 'time'])
        # ce_time = chartevents[['subject_id', 'hadm_id', 'charttime']].drop_duplicates()
        # ce_time['time'] = ce_time.pop('charttime')
        # le_time = labevents[['subject_id', 'hadm_id', 'charttime']].drop_duplicates()
        # le_time['time'] = le_time.pop('charttime')
        # pe_starttime = procedureevents[['subject_id', 'hadm_id', 'starttime']].drop_duplicates()
        # pe_starttime['time'] = pe_starttime.pop('starttime')
        # pe_endtime = procedureevents[['subject_id', 'hadm_id', 'endtime']].drop_duplicates()
        # pe_endtime['time'] = pe_endtime.pop('endtime')
        # all_timestamps = dd.concat([all_timestamps, ce_time, le_time, pe_starttime, pe_endtime]).drop_duplicates().reset_index(drop=True)

        labevents = labevents[labevents["fluid"] == "Blood"]

        self.demographic = demographic.compute()
        self.chartevents = chartevents.compute()
        self.labevents = labevents.compute()
        self.procedureevents = procedureevents.compute()
        self.key_ids = self.key_ids.compute()
        self.all_admissions = all_admissions.compute()
        # self.all_timestamps = all_timestamps.compute()
        self.all_adm_disch = all_adm_disch.compute()

        self.chart_event_list = self.chart_event_list
        # self.lab_event_list = self.labevents[['label', 'fluid']].drop_duplicates().reset_index(drop=True)
        # self.lab_event_list = list(map(lambda x: tuple(x), list(self.lab_event_list.values)))
        self.lab_event_list = self.lab_event_list
        self.procedure_event_list = self.procedure_event_list

        # Encode '---' as NaN
        self.labevents["value"] = replace_str_with_na(
            self.labevents, subset=["value"], by=["label"]
        )

        self.labevents = self.labevents.dropna(axis=0, subset=["value"])

        generate_categorial_num(
            self.demographic, ["insurance", "language", "marital_status", "race"]
        )
        generate_categorial_num(self.chartevents, ["label"])
        # generate_categorial_num(self.labevents, [('label', 'fluid')])
        generate_categorial_num(self.labevents, ["label"])
        generate_categorial_num(self.procedureevents, ["label"])

        generate_age_num(self.demographic)

        self.chartevents.reset_index(drop=True)
        self.labevents.reset_index(drop=True)
        self.procedureevents.reset_index(drop=True)

        self.chartevents["value"] = value_filter(
            self.chartevents,
            by=["label"],
            subset=["value"],
            normal_values=self.normal_values,
        )

        self.chartevents["value_num"] = discretize(
            self.chartevents, subset=["value"], by=["label"]
        )
        self.labevents["value_num"] = discretize(
            self.labevents, subset=["value"], by=["label"]
        )

        self.chartevents["value_num"] = fill_na(
            self.chartevents,
            by=["subject_id", "hadm_id", "label"],
            subset=["value_num"],
        )

        self.demographic.to_csv(
            os.path.join(self.processed_dir, "demographic.csv"), index=False
        )
        self.chartevents.to_csv(
            os.path.join(self.processed_dir, "chartevents.csv"), index=False
        )
        self.labevents.to_csv(
            os.path.join(self.processed_dir, "labevents.csv"), index=False
        )
        self.procedureevents.to_csv(
            os.path.join(self.processed_dir, "procedureevents.csv"), index=False
        )
        self.all_adm_disch.to_csv(
            os.path.join(self.processed_dir, "all_adm_disch.csv"), index=False
        )

    def regenerate_data(self):
        # Generates filtered data according to certain definitions and saves to new subfolder clipped/
        chartevents = dd.read_csv(
            os.path.join(self.data_path, "chartevents.csv"),
            assume_missing=True,
            low_memory=False,
            dtype={"value": "object", "valueuom": "object"},
        )
        labevents = dd.read_csv(
            os.path.join(self.data_path, "labevents.csv"),
            assume_missing=True,
            dtype={
                "storetime": "object",
                "value": "object",
                "valueuom": "object",
                "flag": "object",
                "priority": "object",
                "comments": "object",
            },
        )
        procedureevents = dd.read_csv(
            os.path.join(self.data_path, "procedureevents.csv"),
            assume_missing=True,
            dtype={
                "value": "object",
                "secondaryordercategoryname": "object",
                "totalamountuom": "object",
            },
        )

        chartevents = chartevents[
            chartevents["label"].isin(self.chart_event_list)
        ].reset_index(drop=True)
        labevents = labevents[labevents["label"].isin(self.lab_event_list)].reset_index(
            drop=True
        )
        procedureevents = procedureevents[
            procedureevents["label"].isin(self.procedure_event_list)
        ].reset_index(drop=True)

        chartevents.compute().to_csv(
            os.path.join(self.clipped_dir, "chartevents.csv"), index=False
        )
        labevents.compute().to_csv(
            os.path.join(self.clipped_dir, "labevents.csv"), index=False
        )
        procedureevents.compute().to_csv(
            os.path.join(self.clipped_dir, "procedureevents.csv"), index=False
        )


# if __name__ == '__main__':
#     t1 = perf_counter()
#     x = EHR_dataset(reprocess=False, split='all', task='longstay')
#     for i in range(len(x)):
#         a = x[i]
#     t2 = perf_counter()
#     print(t2-t1)
#     pass
