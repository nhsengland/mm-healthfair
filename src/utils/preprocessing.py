import polars as pl

###############################
# Static data preprocessing
###############################


def transform_gender(data: pl.DataFrame) -> pl.DataFrame:
    """Maps gender values to predefined categories.

    Args:
        data (pl.DataFrame): Data to apply to.

    Returns:
        pl.DataFrame: Updated data.
    """

    g_map = {"F": 1, "M": 0, "OTHER": 2}
    return data.with_columns(gender=pl.col("gender").replace(g_map, default=2))


def transform_marital(data: pl.DataFrame) -> pl.DataFrame:
    """Maps marital status values to predefined categories.

    Args:
        data (pl.DataFrame): Data to apply to.

    Returns:
        pl.DataFrame: Updated data.
    """
    m_map = {"MARRIED": 1, "SINGLE": 2, "WIDOWED": 3, "DIVORCED": 4}
    return data.with_columns(
        marital_status=pl.col("marital_status").replace(m_map, default=0)
    )


def transform_insurance(data: pl.DataFrame) -> pl.DataFrame:
    """Maps insurance status values to predefined categories.

    Args:
        data (pl.DataFrame): Data to apply to.

    Returns:
        pl.DataFrame: Updated data.
    """
    i_map = {"Medicare": 1, "Medicaid": 2, "Other": 0}  # TODO: 0 or nan?
    return data.with_columns(insurance=pl.col("insurance").replace(i_map, default=0))


def transform_race(data: pl.DataFrame) -> pl.DataFrame:
    """Maps race values to predefined categories.

    Args:
        data (pl.DataFrame): Data to apply to.

    Returns:
        pl.DataFrame: Updated data.
    """
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

    # Identifies values with OR
    data = data.with_columns(race=pl.col("race").str.replace(" OR ", "/", literal=True))

    # Strips first value from e.g., White - European
    data = data.with_columns(
        pl.col("race")
        .str.split_exact(" - ", n=1)
        .struct.rename_fields(["race_a", "race_b"])
        .alias("race")
    ).unnest("race")

    # Strips first value from e.g., White / Portuguese
    data = data.with_columns(
        pl.col("race_a")
        .str.split_exact("/", n=1)
        .struct.rename_fields(["race_c", "race_d"])
        .alias("race")
    ).unnest("race")

    data = data.with_columns(race=pl.col("race_c").replace(r_map, default=0)).drop(
        columns=["race_a", "race_b", "race_c", "race_d"]
    )
    return data


def encode_categorical_features(stays: pl.DataFrame) -> pl.DataFrame:
    """Groups and applied one-hot encoding to categorical features.

    Args:
        stays (pl.DataFrame): Stays data.

    Returns:
        pl.DataFrame: Transformed stays data.
    """
    if 'gender' in stays.columns:
        stays = transform_gender(stays)
    if 'race' in stays.columns:
        stays = transform_race(stays)
    if 'marital_status' in stays.columns:
        stays = transform_marital(stays)
    if 'insurance' in stays.columns:
        stays = transform_insurance(stays)

    # apply one-hot encoding to integer columns
    stays = stays.to_dummies([i for i in stays.columns if i in ["gender", "race", "marital_status", "insurance"]])

    return stays


###############################
# Time-series preprocessing
###############################


def clean_events(events: pl.DataFrame) -> pl.DataFrame:
    """Maps non-integer values to None and removes outliers.

    Args:
        events (pl.DataFrame): Events table.

    Returns:
        pl.DataFrame: Cleaned events table.
    """
    # label '__' or "." or "<" or "ERROR" as null value
    # also converts to 2 d.p. floats
    events = events.with_columns(
        value=pl.when(pl.col("value") == ".").then(None).otherwise(pl.col("value"))
    )
    events = events.with_columns(
        value=pl.when(pl.col("value").str.contains("_|<|ERROR"))
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


def add_time_elapsed_to_events(
    events: pl.DataFrame, starttime: pl.Datetime, remove_charttime: bool = False
) -> pl.DataFrame:
    """Adds column 'elapsed' which considers time elapsed since starttime.

    Args:
        events (pl.DataFrame): Events table.
        starttime (pl.Datetime): Reference start time.
        remove_charttime (bool, optional): Whether to remove charttime column. Defaults to False.

    Returns:
        pl.DataFrame: Updated events table.
    """
    events = events.with_columns(
        elapsed=((pl.col("charttime") - starttime) / pl.duration(hours=1)).round(1)
    )

    # reorder columns
    if remove_charttime:
        events = events.drop("charttime")

    return events


def convert_events_to_timeseries(events: pl.DataFrame) -> pl.DataFrame:
    """Converts long-form events to wide-form time-series.

    Args:
        events (pl.DataFrame): Long-form events.

    Returns:
        pl.DataFrame: Wide-form time-series of shape (timestamp, features)
    """

    metadata = (
        events.select(["charttime", "label", "value", "linksto"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime"], keep="last")
        .sort(by="charttime")
    )

    # get unique label, values and charttimes
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )

    # pivot into wide-form format
    timeseries = timeseries.pivot(
        index="charttime", columns="label", values="value"
    ).sort(by="charttime")

    # join any metadata remaining
    timeseries = timeseries.join(
        metadata.select(["charttime", "linksto"]), on="charttime", how="inner"
    )
    return timeseries
