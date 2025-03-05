import os

import numpy as np
import polars as pl

def read_admissions_table(
    mimic4_path: str, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in admissions.csv.gz table and formats column types.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table.
    """
    admits = pl.read_csv(
        os.path.join(mimic4_path, "admissions.csv.gz"),
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
            "insurance",
            "marital_status",
            "race",
            "admission_location",
            "discharge_location"
        ],
        dtypes=[
            pl.Int64,
            pl.Int64,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.String,
            pl.String,
            pl.String,
            pl.String,
            pl.String
        ],
    )
    ### Get eligible admissions with ED attendance and complete sensitive data
    admits = admits.with_columns(
        [
            pl.col("edregtime").is_not_null() & pl.col("edouttime").is_not_null(),
            pl.col("marital_status").is_not_null() & pl.col("race").is_not_null() & pl.col("insurance").is_not_null(),
        ]
    ).filter(
        pl.col("edregtime") < pl.col("admittime")
    ).filter(
        (pl.col("edregtime") < pl.col("dischtime")) & (pl.col("edouttime") < pl.col("dischtime"))
    ).filter(
        (pl.col("edouttime") > pl.col("edregtime")) & (pl.col("admittime") < pl.col("dischtime"))
    ).with_columns(
        (pl.col("dischtime") - pl.col("admittime")).dt.seconds() / (24 * 60 * 60).alias("los_days"),
        (pl.col("los_days") > 7).cast(pl.Int8).alias("ext_stay_7")
    )
    #admits = admits.with_columns(
        #((pl.col("dischtime") - pl.col("admittime")) / pl.duration(days=1)).alias("los")
    #)
    print('Collected admissions table and linked ED attendances..')
    return admits.lazy() if use_lazy else admits

def read_patients_table(
    mimic4_path: str, admissions_data: pl.DataFrame | pl.LazyFrame, 
    use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in patients.csv.gz table and formats column types.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Patients table.
    """
    pats = pl.read_csv(
        os.path.join(mimic4_path, "patients.csv.gz"),
        columns=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        dtypes=[pl.Int64, pl.String, pl.Int64, pl.Int64, pl.Datetime],
    )
    
    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    
    pats = pats.filter(pl.col("subject_id").is_in(admissions_data.select("subject_id").to_series()))
    pats = pats.select(["subject_id", "gender", "dod", "anchor_age", "anchor_year"])
    pats = pats.with_columns(
        (pl.col("anchor_year") - pl.col("anchor_age")).alias("yob")
    ).drop("anchor_year")
    pats = pats.join(admissions_data, on="subject_id", how="left")
    pats = pats.with_columns(
        ((pl.col("dod") < pl.col("dischtime")) & (pl.col("dod") > pl.col("admittime"))).cast(pl.Int8).alias("in_hosp_death"),
        ((~pl.col("discharge_location").str.contains("HOME|DIED|AGAINST ADVICE", literal=True, case=False, na=True)) & (pl.col("in_hosp_death") == 0)).cast(pl.Int8).alias("non_home_discharge")
    )
    print('Collected patients table linked to ED attendances..')
    return pats.lazy() if use_lazy else pats


def read_omr_table(
    mimic4_path: str, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in omr.csv.gz table and formats column types.
    Adds 'los' column based on hospital stay duration.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Omr table.
    """
    omr = pl.read_csv(
        os.path.join(mimic4_path, "omr.csv.gz"),
        dtypes=[pl.Int64, pl.Datetime, pl.Int64, pl.String, pl.String],
    )
    return omr.lazy() if use_lazy else omr


def read_stays_table(
    mimic4_ed_path: str, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in stays.csv.gz table and formats column types.
    Adds 'los_ed' column based on emergency department stay duration.

    Args:
        mimic4_ed_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table.
    """
    stays = pl.read_csv(
        os.path.join(mimic4_ed_path, "edstays.csv.gz"),
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "disposition",
        ],
        dtypes=[pl.Int64, pl.Int64, pl.Int64, pl.Datetime, pl.Datetime, pl.String],
    )
    stays = stays.with_columns(
        ((pl.col("outtime") - pl.col("intime")) / pl.duration(days=1)).alias("los_ed")
    )
    return stays.lazy() if use_lazy else stays


def read_events_table(
    table: str, mimic4_path: str, include_items: list = None
) -> pl.LazyFrame:
    """Reads in events.csv.gz tables from MIMIC-IV and formats column types.

    Args:
        table (str): Name of the events table. Currently supports 'vitalsign' or 'labevents'
        mimic4_path (str): Path to directory containing events
        include_items (list, optional): List of itemid values to filter. Defaults to None.

    Returns:
        pl.LazyFrame : Long-format events table.
    """
    #  Load in csv using polars lazy API (requires table to be in csv format)
    table_df = pl.scan_csv(
        os.path.join(mimic4_path, f"{table}.csv"), try_parse_dates=True
    )

    # add column for linksto
    table_df = table_df.with_columns(linksto=pl.lit(table))

    if "stay_id" not in table_df.columns:
        # add column for stay_id if missing
        table_df = table_df.with_columns(stay_id=pl.lit(None, dtype=pl.Int64))

    if "hadm_id" not in table_df.columns:
        # add column for hadm_id if missing
        table_df = table_df.with_columns(hadm_id=pl.lit(None, dtype=pl.Int64))

    # labevents only
    if table == "labevents":
        d_items = (
            pl.read_csv(os.path.join(mimic4_path, "d_labitems.csv.gz"))
            .lazy()
            .select(["itemid", "label"])
        )

        # merge labitem id's with dict
        table_df = table_df.join(d_items, on="itemid")

        if include_items is not None:
            table_df = table_df.filter(pl.col("itemid").is_in(set(include_items)))

    # for vitalsign need to read/adapt column values
    elif table == "vitalsign":
        vitalsign_column_map = {
            "temperature": "Temperature",
            "heartrate": "Heart rate",
            "resprate": "Respiratory rate",
            "o2sat": "Oxygen saturation",
            "sbp": "Systolic blood pressure",
            "dbp": "Diastolic blood pressure",
        }
        vitalsign_uom_map = {
            "Temperature": "°F",
            "Heart rate": "bpm",
            "Respiratory rate": "insp/min",
            "Oxygen saturation": "%",
            "Systolic blood pressure": "mmHg",
            "Diastolic blood pressure": "mmHg",
        }

        table_df = table_df.rename(vitalsign_column_map)
        table_df = table_df.melt(
            id_vars=["subject_id", "stay_id", "hadm_id", "charttime", "linksto"],
            value_vars=[
                "Temperature",
                "Heart rate",
                "Respiratory rate",
                "Oxygen saturation",
                "Systolic blood pressure",
                "Diastolic blood pressure",
            ],
            variable_name="label",
        ).sort(by="charttime")

        # manually add valueuom
        table_df = table_df.with_columns(
            valueuom=pl.col("label").replace(vitalsign_uom_map)
        )

    else:
        print(f"{table} not yet implemented.")
        raise NotImplementedError

    # select relevant columns
    table_df = table_df.select(
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "value",
            "valueuom",
            "label",
            "linksto",
        ]
    ).cast(
        {
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "stay_id": pl.Int64,
            "charttime": pl.Datetime,
            "value": pl.String,
            "valueuom": pl.String,
            "label": pl.String,
            "linksto": pl.String,
        }
    )

    return table_df


def read_notes(mimic4_path: str, use_lazy: bool = False) -> pl.LazyFrame | pl.DataFrame:
    """Read in discharge summary notes.

    Args:
        mimic4_path (str): _description_
        use_lazy (bool): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: _description_
    """
    notes = pl.read_csv(
        os.path.join(mimic4_path, "discharge.csv.gz"),
        dtypes=[
            pl.String,
            pl.Int64,
            pl.Int64,
            pl.String,
            pl.Int64,
            pl.Datetime,
            pl.Datetime,
            pl.String,
        ],
    ).select(["subject_id", "hadm_id", "charttime", "storetime", "text"])

    return notes.lazy() if use_lazy else notes


def add_omr_variable_to_stays(
    stays: pl.LazyFrame | pl.DataFrame,
    omr: pl.LazyFrame | pl.DataFrame,
    variable: str,
    tolerance: int = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Adds variables from omr table to stays table.

    Args:
        stays (pl.LazyFrame | pl.DataFrame): Stays table.
        omr (pl.LazyFrame | pl.DataFrame): OMR table.
        variable (str): Variable to extract from omr table.
        tolerance (int, optional): Window (days) around admission date to search for data in omr. Defaults to None.

    Returns:
        pl.LazyFrame | pl.DataFrame: Stays table with new variable column.
    """
    # get value of variable on stay/admission date using omr record's date
    # use tolerance to allow elapsed time between dates
    omr = omr.drop("seq_num")

    omr_result_names = (
        omr.select("result_name").collect() if type(omr) == pl.LazyFrame else omr
    )
    omr_result_names = (
        omr_result_names.unique(subset="result_name")
        .get_column("result_name")
        .to_list()
    )

    omr_names = [x for x in omr_result_names if variable.lower() in x.lower()]

    # filter omr readings by variable of interest
    omr = omr.filter(pl.col("result_name").is_in(omr_names))

    # get dates of all stays
    data = stays.with_columns(admitdate=pl.col("admittime").dt.date()).select(
        ["subject_id", "stay_id", "admitdate"]
    )

    # for all the recorded values get column with the admitdate
    data = data.join(omr, how="inner", on="subject_id")

    # filter omr values for when the charttime is within tolerance
    data = data.with_columns(
        chart_diff=((pl.col("admitdate") - pl.col("chartdate")).dt.days()).abs()
    )

    if tolerance is not None:
        data = (
            data.filter(pl.col("chart_diff") <= tolerance)
            .sort(by="chart_diff")
            .drop("chart_diff")
            .unique(subset="stay_id", keep="first")
        )
    else:
        data = (
            data.filter(pl.col("chart_diff") == 0)
            .sort(by="chart_diff")
            .drop("chart_diff")
            .unique(subset="stay_id", keep="first")
        )

    # pivot result_name column
    data = data.melt(
        id_vars=["subject_id", "stay_id"],
        value_name=variable,
        value_vars="result_value",
    ).drop("variable")

    # left join with stays
    stays = stays.join(data, how="left", on=["subject_id", "stay_id"])

    return stays


def add_inhospital_mortality_to_stays(
    stays: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Adds mortality column (binary) to indicate in-hospital mortality.

    Args:
        stays (pl.LazyFrame | pl.DataFrame): Stays table.

    Returns:
        pl.LazyFrame | pl.DataFrame: Stays table with 'mortality' column.
    """
    # If stays table has this column then do not need to manually calculate whether death in hospital has occured.
    if "hospital_expire_flag" in stays.columns:
        stays = stays.rename({"hospital_expire_flag": "mortality"})
    else:
        # Uses dod (date of death), deathtime and admission/discharge time to determine whether in-hospital mortality has occurred.
        stays = stays.with_columns(
            (
                pl.col("dod").is_not_null()
                & (
                    (pl.col("admittime") <= pl.col("dod"))
                    & (pl.col("dischtime") >= pl.col("dod"))
                )
            )
            | (
                pl.col("deathtime").is_not_null()
                & (
                    (pl.col("admittime") <= pl.col("deathtime"))
                    & (pl.col("dischtime") >= pl.col("deathtime"))
                )
            )
            .cast(pl.UInt8)
            .alias("mortality")
        )
    return stays


def get_hadm_id_from_admits(
    events: pl.LazyFrame | pl.DataFrame, admits: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame | pl.DataFrame:
    """Uses admissions table to extract hadm_id based on events charttime.

    Args:
        events (pl.LazyFrame | pl.DataFrame): Events table.
        admits (pl.LazyFrame | pl.DataFrame): Admissions table.

    Returns:
        pl.LazyFrame | pl.DataFrame: Events table with filled hadm_id values where possible.
    """
    # get the hadm_id and admission/discharge time window
    admits = admits.select(["subject_id", "hadm_id", "admittime", "dischtime"])

    # for each charted event, join with subject-level admissions
    data = events.select(["charttime", "label", "subject_id"]).join(
        admits, how="inner", on="subject_id"
    )

    # filter by whether charttime is between admittime and dischtime
    data = data.filter(pl.col("charttime").is_between("admittime", "dischtime")).select(
        ["subject_id", "hadm_id", "charttime", "label"]
    )

    # now add hadm_id to df by charttime and subject_id
    events = events.join(
        data.unique(["subject_id", "hadm_id", "charttime", "label"]),
        on=["subject_id", "label", "charttime"],
        how="left",
    )

    # fill missing values where possible
    events = events.with_columns(
        hadm_id=pl.when(pl.col("hadm_id").is_null())
        .then(pl.col("hadm_id_right"))
        .otherwise(pl.col("hadm_id"))
    ).drop("hadm_id_right")
    return events


def filter_on_nb_stays(
    stays: pl.LazyFrame | pl.DataFrame, min_nb_stays: int = 1, max_nb_stays: int = 1
) -> pl.LazyFrame | pl.DataFrame:
    """Filters stays to ensure certain number of emergency department stays per hospital admission (typically 1).

    Args:
        stays (pl.LazyFrame | pl.DataFrame): Stays table.
        min_nb_stays (int, optional): Minimum number of stays per admission. Defaults to 1.
        max_nb_stays (int, optional): Maximum number of stays per admission. Defaults to 1.

    Returns:
        pl.LazyFrame | pl.DataFrame: Filtered stays table.
    """
    # Only keep hospital admissions that are associated with a certain number of ED stays (within min and max)
    to_keep = stays.group_by("hadm_id").agg(pl.col("stay_id").count())
    to_keep = to_keep.filter(
        (pl.col("stay_id") >= min_nb_stays) & (pl.col("stay_id") <= max_nb_stays)
    ).select("hadm_id")
    stays = stays.join(to_keep, how="inner", on="hadm_id")
    return stays


def filter_stays_on_age(
    stays: pl.LazyFrame | pl.DataFrame, min_age=18, max_age=np.inf
) -> pl.LazyFrame | pl.DataFrame:
    """Filter stays based on patient age.

    Args:
        stays (pl.LazyFrame | pl.DataFrame): Stays table.
        min_age (int, optional): Minimum patient age. Defaults to 18.
        max_age (_type_, optional): Maximum patient age. Defaults to np.inf.

    Returns:
        pl.LazyFrame | pl.DataFrame: Filtered stays table.
    """
    # must have already added age to stays table
    stays = stays.filter(
        (pl.col("anchor_age") >= min_age) & (pl.col("anchor_age") <= max_age)
    )
    return stays
