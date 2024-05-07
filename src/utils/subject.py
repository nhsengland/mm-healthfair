# Code adapted from

import os

import numpy as np
import polars as pl

from .mimiciv import read_admissions_table


def read_stays(subject_path):
    stays = pl.scan_csv(os.path.join(subject_path, "stays.csv"))
    stays = stays.cast(
        {
            "subject_id": pl.Int64,
            "intime": pl.Datetime,
            "outtime": pl.Datetime,
            "admittime": pl.Datetime,
            "dischtime": pl.Datetime,
            "dod": pl.Datetime,
        }
    )
    stays = stays.sort(by=["intime", "outtime"])
    return stays


def read_diagnoses(subject_path):
    return pl.scan_csv(os.path.join(subject_path, "diagnoses.csv"))


def read_events(subject_path, mimic4_path, drop_na_hadm_stay=True):
    events = pl.scan_csv(
        os.path.join(subject_path, "events.csv"), null_values="___"
    ).unique()

    events = events.cast(
        {"subject_id": pl.Int64, "stay_id": pl.Int64, "hadm_id": pl.Int64}
    )
    events = events.with_columns(
        charttime=pl.col("charttime").str.strptime(pl.Datetime, strict=False)
    )

    stays = read_stays(os.path.join(subject_path))
    # use admissions table to impute missing hadm_ids based on charttime
    admits = read_admissions_table(mimic4_path, use_lazy=True)

    events = get_hadm_id_from_admits(events, admits)

    # then use transform to apply function to fill in missing hadm_id or stay_id from stay table
    events = impute_from_df(events, stays, "stay_id", "hadm_id")
    events = impute_from_df(events, stays, "hadm_id", "stay_id")

    # if both hadm_id and stay_id still can't be found then drop row
    if drop_na_hadm_stay:
        events = events.filter(
            (pl.col("hadm_id").is_not_null()) & (pl.col("stay_id").is_not_null())
        )

    # for remaining rows with only one ID mark with -1 (for identification later / quality checking)
    events = events.with_columns(pl.col("stay_id").fill_null(value=-1))
    events = events.with_columns(pl.col("hadm_id").fill_null(value=-1))
    return events


def impute_from_df(
    impute_to,
    impute_from,
    use_col: str = None,
    key_col: str = None,
):
    dict_map = (
        impute_from.select([key_col, use_col])
        .collect()
        .rows_by_key(key=key_col, unique=True)
    )
    impute_to = impute_to.with_columns(pl.col(key_col).replace(dict_map).alias(use_col))

    return impute_to


def get_hadm_id_from_admits(df, admits):
    # similar to impute from df but using a range instead of exact value match

    # get the hadm_id and admission/discharge time window
    admits = admits.select(["subject_id", "hadm_id", "admittime", "dischtime"])

    # for each charted event, join with subject-level admissions
    data = df.select(["charttime", "label", "subject_id"]).join(
        admits, how="inner", on="subject_id"
    )

    # filter by whether charttime is between admittime and dischtime
    data = data.filter(pl.col("charttime").is_between("admittime", "dischtime")).select(
        ["subject_id", "hadm_id", "charttime", "label"]
    )

    # now add hadm_id to df by charttime and subject_id
    df = df.join(
        data.unique(["subject_id", "hadm_id", "charttime", "label"]),
        on=["subject_id", "label", "charttime"],
        how="left",
    )

    # fill missing values where possible
    df = df.with_columns(
        hadm_id=pl.when(pl.col("hadm_id").is_null())
        .then(pl.col("hadm_id_right"))
        .otherwise(pl.col("hadm_id"))
    ).drop("hadm_id_right")
    return df


def get_events_in_period(events, stay_id, hadm_id, starttime=None, endtime=None):
    if starttime is not None and endtime is not None:
        # also filter so events must fall within start and end time
        events = events.filter(pl.col("charttime").is_between(starttime, endtime))
    else:
        # get events linked to ED stay or hosp admission
        events = events.filter(
            ((pl.col("stay_id") == stay_id) & (pl.col("linksto") == "vitalsign"))
            | ((pl.col("hadm_id") == hadm_id) & (pl.col("linksto") == "labevents"))
        )

    events = events.drop(["stay_id", "hadm_id", "linksto"])
    return events


def add_hours_elapsed_to_events(events, time, remove_charttime=True):
    events = events.with_columns(
        hours=((pl.col("charttime") - time) / pl.duration(hours=1)).round(1)
    )

    if remove_charttime:
        events = events.drop("charttime")
    return events


def convert_events_to_timeseries(events):
    metadata = (
        events.select(["charttime", "hadm_id", "stay_id", "linksto"])
        .sort(by=["charttime", "hadm_id", "stay_id"])
        .unique(keep="first")
    )
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )
    timeseries = (
        timeseries.collect()
        .pivot(index="charttime", columns="label", values="value")
        .lazy()
        .join(metadata, on="charttime")
        .sort(by="charttime")
    )
    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
