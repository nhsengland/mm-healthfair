import numpy as np
import polars as pl

###############################
# Non-time series preprocessing
###############################


def transform_gender(data):
    g_map = {"F": 1, "M": 2, "OTHER": 3}
    return data.with_columns(gender=pl.col("gender").replace(g_map, default=0))


def transform_marital(data):
    m_map = {"MARRIED": 1, "SINGLE": 2, "WIDOWED": 3, "DIVORCED": 4}
    return data.with_columns(
        marital_status=pl.col("marital_status").replace(m_map, default=0)
    )


def transform_insurance(data):
    i_map = {"Medicare": 1, "Medicaid": 2, "Other": 3}  # TODO: 0 or nan?
    return data.with_columns(insurance=pl.col("insurance").replace(i_map, default=0))


def transform_race(race_series):
    r_map = {
        "ASIAN": 1,
        "BLACK": 2,
        "CARIBBEAN ISLAND": 2,
        "HISPANIC": 3,
        "SOUTH AMERICAN": 3,
        "WHITE": 4,
        "MIDDLE EASTERN": 4,
        "PORTUGUESE": 4,
        "AMERICAN INDIAN": 0,
        "NATIVE HAWAIIAN": 0,
        "UNABLE TO OBTAIN": 0,
        "PATIENT DECLINED TO ANSWER": 0,
        "UNKNOWN": 0,
        "OTHER": 0,
    }

    race_series = race_series.with_columns(
        race=pl.col("race").str.replace(" OR ", "/", literal=True)
    )
    race_series = race_series.with_columns(
        pl.col("race")
        .str.split_exact(" - ", n=1)
        .struct.rename_fields(["race_a", "race_b"])
        .alias("race")
    ).unnest("race")
    race_series = race_series.with_columns(
        pl.col("race_a")
        .str.split_exact("/", n=1)
        .struct.rename_fields(["race_c", "race_d"])
        .alias("race")
    ).unnest("race")

    race_series = race_series.with_columns(
        race=pl.col("race_c").replace(r_map, default=0)
    ).drop(columns=["race_a", "race_b", "race_c", "race_d"])
    return race_series


def impute_from_df(
    impute_to,
    impute_from,
    use_col: str = None,
    key_col: str = None,
):
    dict_map = (
        impute_from.select([key_col, use_col])
        .collect()
        .rows_by_key(key=use_col, unique=True)
    )

    impute_to = impute_to.with_columns(new=pl.col(use_col).replace(dict_map))
    impute_to = impute_to.with_columns(
        pl.when(pl.col("hadm_id").is_null())
        .then(pl.col("new"))
        .otherwise(pl.col("hadm_id"))
        .alias(key_col)
    ).drop("new")

    return impute_to


def process_demographic_data(stays, features=None):
    # TODO: Normalise height, weight and age

    stays = (
        stays.select(["hadm_id"] + features)
        .cast({"los": pl.Float64, "los_ed": pl.Float64})
        .collect()
    )

    stays = transform_gender(stays)
    stays = transform_race(stays)
    stays = transform_marital(stays)
    stays = transform_insurance(stays)

    # Round all float values to 1.dp
    stays = stays.with_columns(pl.selectors.by_dtype(pl.FLOAT_DTYPES).round(1))

    return stays


def map_itemids_to_variables(events, var_map):
    return events.join(var_map, on="itemid")


def clean_events(events):
    # label '__' as null value
    # also converts to 2 d.p. floats
    events = events.with_columns(
        value=pl.when(pl.col("value").str.contains("_"))
        .then(None)
        .otherwise(pl.col("value"))
        .cast(pl.Float64)
    )

    # Remove outliers using 2 std from mean
    events = events.with_columns(mean=pl.col("value").mean().over(pl.count("label")))
    events = events.with_columns(std=pl.col("value").std().over(pl.count("label")))
    events = events.filter(
        (pl.col("value") < pl.col("mean") + pl.col("std") * 2)
        & (pl.col("value") > pl.col("mean") - pl.col("std") * 2)
    ).drop(["mean", "std"])
    return events


def add_time_elapsed_to_events(events, remove_charttime=False):
    events = events.with_columns(
        elapsed=(
            (pl.col("charttime") - pl.col("charttime").first()) / pl.duration(hours=1)
        ).round(1)
    )

    # reorder columns
    if remove_charttime:
        events = events.drop("charttime")

    return events


def convert_events_to_timeseries(events):
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )
    timeseries = timeseries.pivot(
        index="charttime", columns="label", values="value"
    ).sort(by="charttime")
    return timeseries


def get_events_in_period(events, stay_id, hadm_id, starttime=None, endtime=None):
    if starttime is not None and endtime is not None:
        # also filter so events must fall within start and end time
        events = events.filter(pl.col("charttime").is_between(starttime, endtime))

    # get events linked to ED stay or hosp admission
    events = events.filter(
        (pl.col("stay_id") == stay_id) | (pl.col("hadm_id") == hadm_id)
    )

    events = events.drop(["stay_id", "hadm_id", "linksto"])
    return events


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
