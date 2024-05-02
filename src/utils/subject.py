# Code adapted from

import os

import numpy as np
import pandas as pd

from .functions import dataframe_from_csv
from .mimiciv import read_admissions_table


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


def read_events(subject_path, mimic4_path, drop_na_hadm_stay=True):
    events = dataframe_from_csv(os.path.join(subject_path, "events.csv"))
    stays = dataframe_from_csv(os.path.join(subject_path, "stays.csv"))

    events.charttime = pd.to_datetime(events.charttime, format="ISO8601")
    events.valueuom = events.valueuom.fillna("").astype(str)

    # use admissions table to impute missing hadm_ids based on charttime
    admits = read_admissions_table(mimic4_path)
    events["hadm_id"] = events.apply(
        lambda row: get_hadm_id_from_admits(row, admits)
        if pd.isnull(row["hadm_id"])
        else row["hadm_id"],
        axis=1,
    )

    # then use transform to apply function to fill in missing hadm_id or stay_id from stay table
    events.hadm_id = events.apply(
        lambda row: impute_missing(
            impute_to=row, impute_from=stays, use_col="stay_id", value_col="hadm_id"
        )
        if pd.isnull(row.hadm_id)
        else int(row.hadm_id),
        axis=1,
    )
    events.stay_id = events.apply(
        lambda row: impute_missing(
            impute_to=row, impute_from=stays, use_col="hadm_id", value_col="stay_id"
        )
        if ("stay_id" not in row) or pd.isnull(row.stay_id)
        else int(row.stay_id),
        axis=1,
    )

    # if both hadm_id and stay_id still can't be found then drop row
    if drop_na_hadm_stay:
        events = events.dropna(subset=["hadm_id", "stay_id"], how="all")

    # for remaining rows with only one ID mark with -1 (for identification later / quality checking)
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)

    # events.sort_values(by=['charttime', 'itemid', 'stay_id'], inplace=True)
    return events


def impute_missing(
    impute_to: pd.DataFrame = None,
    impute_from: pd.DataFrame = None,
    use_col: str = None,
    value_col: str = None,
    cannot_impute_val=np.nan,
):
    idx = impute_from[use_col] == impute_to[use_col]  # find corresponding stay
    if idx.values[0] is True:  # if one can be found
        return impute_from[idx][value_col]  # get value
    else:
        return cannot_impute_val  # fill na with missing val if cannot be matched


def get_hadm_id_from_admits(row, admits, time_col="charttime"):
    # create bool
    idx = (
        (admits["admittime"] <= row[time_col])
        & (row[time_col] <= admits["dischtime"])
        & (admits["subject_id"] == row["subject_id"])
    )

    if any(idx) is True:
        # apply bool and returns hadm_id value
        return admits[idx].hadm_id.values[0]
    else:
        return np.nan


def get_events_in_period(events, stay_id, hadm_id, starttime=None, endtime=None):
    # get events linked to ED stay
    vitalsign_idx = (events.stay_id == stay_id) & (events.linksto == "vitalsign")
    # get all events linked to HADM stay
    labevents_idx = (events.hadm_id == hadm_id) & (events.linksto == "labevents")

    if starttime is not None and endtime is not None:
        # also filter so events must fall within start and end time
        vitalsign_idx = (
            vitalsign_idx
            & (events.charttime >= starttime)
            & (events.charttime <= endtime)
        )
        labevents_idx = (
            labevents_idx
            & (events.charttime >= starttime)
            & (events.charttime <= endtime)
        )

    events = events[vitalsign_idx | labevents_idx]
    del events["stay_id"]
    del events["hadm_id"]
    del events["linksto"]
    return events


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events["hours"] = (
        (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, "s")) / 60.0 / 60
    )
    if remove_charttime:
        del events["charttime"]
    return events


def convert_events_to_timeseries(events):
    metadata = (
        events[["charttime", "hadm_id", "stay_id", "linksto"]]
        .sort_values(by=["charttime", "hadm_id", "stay_id"])
        .drop_duplicates(keep="first")
        .set_index("charttime")
    )
    timeseries = (
        events[["charttime", "label", "value"]]
        .sort_values(by=["charttime", "label", "value"], axis=0)
        .drop_duplicates(subset=["charttime", "label"], keep="last")
    )
    timeseries = (
        timeseries.pivot(index="charttime", columns="label", values="value")
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
