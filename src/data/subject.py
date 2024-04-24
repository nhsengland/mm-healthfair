# Code adapted from

import os

import numpy as np
import pandas as pd

from .util import dataframe_from_csv


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, "stays.csv"))
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.admittime = pd.to_datetime(stays.admittime)
    stays.dischtime = pd.to_datetime(stays.dischtime)
    stays.sort_values(by=["intime", "outtime"], inplace=True)
    return stays


def read_diagnoses(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, "diagnoses.csv"))


def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, "events.csv"))
    stays = dataframe_from_csv(os.path.join(subject_path, "stays.csv"))
    if remove_null:
        events = events[events.value.notnull()]
    events.charttime = pd.to_datetime(events.charttime)
    events.valueuom = events.valueuom.fillna("").astype(str)
    # use transform to apply function to fill in missing hadm_id or stay_id from stays
    events.hadm_id = events.apply(
        lambda row: impute_missing_id(row.stay_id, stays)
        if np.isnan(row.hadm_id)
        else row.hadm_id,
        axis=1,
    ).astype(int)
    events.stay_id = events.apply(
        lambda row: impute_missing_id(row.hadm_id, stays, impute_col="stay_id")
        if ("stay_id" not in row) or np.isnan(row.stay_id)
        else row.stay_id,
        axis=1,
    ).astype(int)
    # events.stay_id = events.stay_id.fillna(value=-1).astype(int)
    # events.sort_values(by=['charttime', 'itemid', 'stay_id'], inplace=True)
    return events


def impute_missing_id(id, stays, impute_col="hadm_id", missing_val=-1):
    if impute_col == "hadm_id":
        idx = stays.stay_id == id  # find corresponding stay
        if idx.values[0] is True:  # if one can be found
            return stays[idx].hadm_id  # get hadm_id
        else:
            return missing_val  # fill na with missing val
    else:
        idx = stays.hadm_id == id
        if idx.values[0] is True:
            return stays[idx].stay_id
        else:
            return missing_val


def get_events_for_stay(events, stay_id, intime=None, outtime=None):
    idx = events.stay_id == stay_id
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))
    events = events[idx]
    del events["stay_id"]
    return events


def get_events_for_admission(events, hadm_id, admittime=None, dischtime=None):
    idx = events.hadm_id == hadm_id
    if admittime is not None and dischtime is not None:
        idx = idx | ((events.charttime >= admittime) & (events.charttime <= dischtime))
    events = events[idx]
    del events["hadm_id"]
    return events


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events["hours"] = (
        (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, "s")) / 60.0 / 60
    )
    if remove_charttime:
        del events["charttime"]
    return events


def convert_events_to_timeseries(events, variable_column="itemid"):
    metadata = (
        events[["charttime", "hadm_id", "stay_id"]]
        .sort_values(by=["charttime", "hadm_id", "stay_id"])
        .drop_duplicates(keep="first")
        .set_index("charttime")
    )
    timeseries = (
        events[["charttime", variable_column, "value"]]
        .sort_values(by=["charttime", variable_column, "value"], axis=0)
        .drop_duplicates(subset=["charttime", variable_column], keep="last")
    )
    timeseries = (
        timeseries.pivot(index="charttime", columns=variable_column, values="value")
        .merge(metadata, left_index=True, right_index=True)
        .sort_index(axis=0)
        .reset_index()
    )

    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
