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
    pats = pats.filter(pl.col("anchor_age") >= 18)
    pats = pats.with_columns(
        ((pl.col("dod") < pl.col("dischtime")) & (pl.col("dod") > pl.col("admittime"))).cast(pl.Int8).alias("in_hosp_death"),
        ((~pl.col("discharge_location").str.contains("HOME|DIED|AGAINST ADVICE", literal=True, case=False, na=True)) & (pl.col("in_hosp_death") == 0)).cast(pl.Int8).alias("non_home_discharge")
    )
    print('Collected patients table linked to ED attendances..')
    return pats.lazy() if use_lazy else pats

def read_icu_table(
    mimic4_ed_path: str, admissions_data: pl.DataFrame | pl.LazyFrame, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in icustays.csv.gz table and parses ICU admission outcome.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: ICU stays table.
    """
    icu = pl.read_csv(
        os.path.join(mimic4_ed_path, "icu/icustays.csv.gz"),
        columns=['subject_id', 'hadm_id', 'intime', 'outtime', 'los'],
        dtypes=[pl.Int64, pl.Int64, pl.Datetime, pl.Datetime, pl.Int16]
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    print("Original number of ICU stays:", icu.shape[0], icu.select("subject_id").n_unique())
    icu = icu.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id")) &
        pl.col("hadm_id").is_in(admissions_data.select("hadm_id"))
    )
    print("Number of ICU stays with validated ED attendances:", icu.shape[0], icu.select("subject_id").n_unique())
    icu_eps = admissions_data.join(icu, on=["subject_id", "hadm_id"], how="left").sort(
        by=["subject_id", "hadm_id", "intime"]
    ).unique(subset=["subject_id", "hadm_id"], keep="last")
    icu_eps = icu_eps.with_columns(
        (pl.col("intime") > pl.col("admittime") & pl.col("outtime") < pl.col("dischtime")).cast(pl.Int8).alias("icu_admission"),
        pl.col("los").alias("icu_los_days")
    )
    print('Collected ICU stay outcomes..')
    return icu_eps.lazy() if use_lazy else icu_eps

def read_d_icd_diagnoses_table(mimic4_path):
    d_icd = pl.read_csv(
        os.path.join(mimic4_path, 'd_icd_diagnoses.csv.gz'),
        columns=['icd_code', 'long_title'],
        dtypes=[pl.String, pl.String]
    )
    return d_icd

def read_diagnoses_table(
    mimic4_path: str, admissions_data: pl.DataFrame | pl.LazyFrame, 
    adm_last: pl.DataFrame | pl.LazyFrame, 
    use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in diagnoses_icd.csv.gz table and formats column types.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Diagnoses table.
    """
    diag = pl.read_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz"),
        columns=["subject_id", "hadm_id", "seq_num", "icd_code"],
        dtypes=[pl.Int64, pl.Int64, pl.Int16, pl.String],
    )
    diag_mapping = read_d_icd_diagnoses_table(mimic4_path)
    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(adm_last, pl.LazyFrame):
        adm_last = adm_last.collect()

    diag = diag.join(diag_mapping, on='icd_code', how='inner')
    print("Original number of diagnoses:", diag.shape[0], diag.select(pl.col("subject_id").n_unique()))
    
    # Get list of eligible hospital episodes as historical data
    adm_lkup = admissions_data.filter(~pl.col('hadm_id').is_in(adm_last.select('hadm_id')))
    adm_lkup = adm_lkup.join(
        adm_last.select(['subject_id', 'edregtime']).rename({'edregtime': 'last_edregtime'}),
        on='subject_id',
        how='left'
    )
    adm_lkup = adm_lkup.filter(pl.col('edregtime') < pl.col('last_edregtime'))
    
    # Filter diagnoses for lookup episodes
    diag = diag.filter(pl.col('subject_id').is_in(adm_lkup.select('subject_id')))
    diag = diag.filter(pl.col('hadm_id').is_in(adm_lkup.select('hadm_id')))

    print('Collected diagnoses table..')
    return diag.lazy() if use_lazy else diag


def read_notes(admissions_data: pl.DataFrame | pl.LazyFrame, 
               admits_last: pl.DataFrame | pl.LazyFrame,
               mimic4_path: str, verbose: bool = True,
               use_lazy: bool = False) -> pl.LazyFrame | pl.DataFrame:
    """Read in discharge summary and preprocessed BHC segments.

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
            pl.String
        ],
    ).select(["subject_id", "hadm_id", "charttime", "storetime", "text"])

    notes_ext = pl.read_csv(
        os.path.join(mimic4_path, "mimic-iv-bhc.csv"),
        dtypes=[
            pl.String,
            pl.String,
            pl.String,
            pl.Int64,
            pl.Int64
        ],
    ).select(['note_id', 'input', 'target', 'input_tokens', 'target_tokens'])
    
    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Merge with ED attendances cohort
    if verbose:
        print("Original number of notes:", notes.shape[0], notes.select("subject_id").n_unique())
    notes = notes.filter(pl.col("subject_id").is_in(admissions_data.select("subject_id").to_series()))
    if verbose:
        print("Number of notes with validated ED attendances:", notes.shape[0], notes.select("subject_id").n_unique())
    notes = notes.join(notes_ext, how='left', on='note_id')
    if verbose:
        print("Number of total matching preprocessed notes:", notes.filter(pl.col("input").is_not_null()).shape[0], notes.filter(pl.col("target").is_not_null()).shape[0])
        print("Unique patients with matching preprocessed notes:", notes.filter(pl.col("input").is_not_null()).select("subject_id").n_unique(), 
              notes.filter(pl.col("target").is_not_null()).select("subject_id").n_unique())
    
    adm_notes = admissions_data.select(["subject_id", "hadm_id", "admittime"]).join(
        notes.select(["note_id", "subject_id", "hadm_id", "charttime", "text", "input", "target", "input_tokens", "target_tokens"]),
        on=["subject_id", "hadm_id"],
        how="left"
    ).filter(pl.col("target").is_not_null())
    
    ### Get previous hospital episodes as historical data
    adm_lkup = adm_notes.join(
        admits_last.select(["subject_id", "edregtime"]).rename({"edregtime": "last_edregtime"}),
        on="subject_id",
        how="left"
    ).filter(pl.col("edregtime") < pl.col("last_edregtime"))
    
    ### Get full notes history for each eligible patient
    adm_notes = adm_notes.filter(pl.col("hadm_id").is_in(adm_lkup.select("hadm_id").to_series()))
    
    if verbose:
        print("Unique patients with matching preprocessed notes:", notes.filter(pl.col("input").is_not_null()).select("subject_id").n_unique(), 
              notes.filter(pl.col("target").is_not_null()).select("subject_id").n_unique())
        print('Min, Mean and Max historical notes per patient:', adm_notes.groupby("subject_id").agg(pl.count("note_id").min()).collect()[0, 0],
              adm_notes.groupby("subject_id").agg(pl.count("note_id").mean()).collect()[0, 0],
              adm_notes.groupby("subject_id").agg(pl.count("note_id").max()).collect()[0, 0])
    
    return adm_notes.lazy() if use_lazy else adm_notes

def get_notes_population(adm_notes: pl.DataFrame | pl.LazyFrame, 
                         admit_last: pl.DataFrame | pl.LazyFrame, 
                         use_lazy: bool = False) -> pl.DataFrame:
    """Gets population of unique ED patients with existing note history."""
    if isinstance(adm_notes, pl.LazyFrame):
        adm_notes = adm_notes.collect()
    if isinstance(admit_last, pl.LazyFrame):
        admit_last = admit_last.collect()

    ### Aggregate historical notes data with demographics
    notes_grouped = adm_notes.groupby('subject_id').agg([
        pl.col('hadm_id').n_unique().alias('num_summaries'),
        pl.col('input_tokens').sum().alias('num_input_tokens'),
        pl.col('target_tokens').sum().alias('num_target_tokens'),
        pl.col('target').apply(lambda x: ' <ENDNOTE> <STARTNOTE> '.join(x)).alias('target')
    ])
    ### Filter population with at least one note
    ed_pts = admit_last.filter(pl.col('subject_id').is_in(notes_grouped.select('subject_id')))
    ## Save number of tokens per patient
    ed_pts = ed_pts.join(notes_grouped.select(['subject_id', 'num_input_tokens', 'num_target_tokens']), on='subject_id', how='left')
    
    return ed_pts.lazy() if use_lazy else ed_pts, notes_grouped.lazy() if use_lazy else notes_grouped

def read_omr_table(
    mimic4_path: str, admits_last: pl.DataFrame | pl.LazyFrame, 
    verbose: bool = True, use_lazy: bool = False,
    vitalsign_uom_map: dict = {
            "Temperature": "°F",
            "Heart rate": "bpm",
            "Respiratory rate": "insp/min",
            "Oxygen saturation": "%",
            "Systolic blood pressure": "mmHg",
            "Diastolic blood pressure": "mmHg",
            "BMI": "kg/m²"
        }
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in omr.csv.gz table and formats column types.
    Sets measures for blood pressure and BMI in long format.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Omr table.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    omr = pl.read_csv(
        os.path.join(mimic4_path, "omr.csv.gz"),
        dtypes=[pl.Int64, pl.Datetime, pl.Int64, pl.String, pl.String],
    )
    omr = omr.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    omr = omr.with_columns(pl.col("chartdate").str.strptime(pl.Date, "%Y-%m-%d").alias("charttime"))
    omr = omr.drop("seq_num")

    ### Prepare hospital measures time-series
    omr = omr.join(admits_last.select(["subject_id", "edregtime"]), on="subject_id", how="left")
    omr = omr.filter(pl.col("charttime") <= pl.col("edregtime"))
    omr = omr.drop("edregtime")
    omr = omr.with_columns(
        pl.when(pl.col("result_name").str.contains("Blood Pressure")).then("bp").otherwise(pl.col("result_name")).alias("result_name"),
        pl.col("result_value").str.split("/").arr.get(0).cast(pl.Float64).alias("result_sysbp"),
        pl.col("result_value").str.split("/").arr.get(1).cast(pl.Float64).alias("result_diabp"),
        pl.when(pl.col("result_name").str.contains("BMI")).then("bmi").otherwise(pl.col("result_name")).alias("result_name")
    )

    # Create separate rows for sysbp and diabp
    sysbp_measures = omr.select(["subject_id", "charttime", "result_sysbp"]).rename({"result_sysbp": "value"}).with_columns(pl.lit("Systolic blood pressure").alias("label"))
    diabp_measures = omr.select(["subject_id", "charttime", "result_diabp"]).rename({"result_diabp": "value"}).with_columns(pl.lit("Diastolic blood pressure").alias("label"))

    # Concatenate the sysbp and diabp measures
    bp_measures = sysbp_measures.vstack(diabp_measures)
    # Add BMI measurements
    bmi_measures = omr.filter(pl.col("result_name") == "bmi").select(["subject_id", "charttime", "result_value"]).rename({"result_value": "value"}).with_columns(pl.lit("BMI").alias("label"))
    omr = bp_measures.vstack(bmi_measures)

    # Map the value_uom
    omr = omr.with_columns(pl.col("label").map_dict(vitalsign_uom_map).alias("value_uom"))

    return omr.lazy() if use_lazy else omr

def read_vitals_table(
    mimic4_ed_path: str, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Reads in vitalsign.csv.gz table and formats column types.

    Args:
        mimic4_path (str): Path to directory containing downloaded MIMIC-IV hosp module files.
        use_lazy (bool, optional): Whether to return a Polars LazyFrame or DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame | pl.DataFrame: Vitals table.
    """
    vitals = pl.read_csv(
        os.path.join(mimic4_path, "vitalsign.csv.gz"),
        dtypes=[pl.Int64, pl.Int64, pl.Datetime, pl.String, pl.String, pl.String],
    )
    return vitals.lazy() if use_lazy else vitals


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
