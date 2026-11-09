import os

import numpy as np
import pandas as pd
import polars as pl

from utils.preprocessing import (
    clean_labevents,
    prepare_medication_features,
    rename_fields,
    transform_sensitive_attributes,
)


def read_admissions_table(
    mimic4_path: str,
    use_lazy: bool = False,
    verbose: bool = True,
    ext_stay_threshold: int = 7,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the admissions table from MIMIC-IV, setting up the ED population.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV hospital module files.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.
        ext_stay_threshold (int): Threshold (in days) for setting extended stay outcome.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table with additional columns.
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
            "discharge_location",
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
            pl.String,
        ],
        try_parse_dates=True,
    )

    admits = admits.filter(
        pl.col("edregtime").is_not_null()
        & pl.col("edouttime").is_not_null()
        & pl.col("marital_status").is_not_null()
        & pl.col("race").is_not_null()
        & pl.col("insurance").is_not_null()
    )
    if verbose:
        print(
            "Number of admissions with complete data:",
            admits.shape[0],
            admits.select("subject_id").n_unique(),
        )

    admits = admits.filter(
        (pl.col("edregtime") < pl.col("admittime"))
        & (pl.col("edregtime") < pl.col("dischtime"))
        & (pl.col("edouttime") < pl.col("dischtime"))
        & (pl.col("edouttime") > pl.col("edregtime"))
        & (pl.col("admittime") < pl.col("dischtime"))
    )
    if verbose:
        print(
            "Number of admissions after timestamp validation:",
            admits.shape[0],
            admits.select("subject_id").n_unique(),
        )

    admits = admits.with_columns(
        (
            (pl.col("dischtime") - pl.col("admittime")).dt.seconds() / (24 * 60 * 60)
        ).alias("los_days")
    )

    admits = admits.with_columns(
        (pl.col("los_days") > ext_stay_threshold).cast(pl.Int8).alias("ext_stay_7")
    )
    if verbose:
        print(
            f"Subjects with extended stay > 7 days: {admits.filter(pl.col('ext_stay_7') == 1).select('subject_id').n_unique()}, % of pts: {admits.filter(pl.col('ext_stay_7') == 1).select('subject_id').n_unique() / admits.select('subject_id').n_unique() * 100:.2f}"
        )

    print("Collected admissions table and linked ED attendances..")
    return admits.lazy() if use_lazy else admits


def read_patients_table(
    mimic4_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    age_cutoff: int = 18,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the patients table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        age_cutoff (int): Minimum age to include.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: Patients table with joined admissions and derived outcomes.
    """
    pats = pl.read_csv(
        os.path.join(mimic4_path, "patients.csv.gz"),
        columns=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        dtypes=[pl.Int64, pl.String, pl.Int64, pl.Int64, pl.Datetime],
        try_parse_dates=True,
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    pats = pats.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id").to_series())
    )
    pats = pats.select(["subject_id", "gender", "dod", "anchor_age", "anchor_year"])
    pats = pats.with_columns(
        (pl.col("anchor_year") - pl.col("anchor_age")).alias("yob")
    ).drop("anchor_year")
    pats = pats.join(admissions_data, on="subject_id", how="left")
    pats = pats.filter(pl.col("anchor_age") >= age_cutoff)
    pats = pats.with_columns(pl.col("discharge_location").fill_null("UNKNOWN"))
    pats = pats.with_columns(
        ((pl.col("dod") <= pl.col("dischtime")) & (pl.col("dod") > pl.col("admittime")))
        .cast(pl.Int8)
        .alias("in_hosp_death")
    )
    pats = pats.with_columns(
        (
            (
                ~pl.col("discharge_location").is_in(
                    [
                        "HOME",
                        "HOME HEALTH CARE",
                        "DIED",
                        "AGAINST ADVICE",
                        "ASSISTED LIVING",
                        "UNKNOWN",
                    ]
                )
            )
            & (pl.col("in_hosp_death") == 0)
        )
        .cast(pl.Int8)
        .alias("non_home_discharge")
    )
    pats = pats.with_columns(
        pl.col("in_hosp_death").fill_null(0).cast(pl.Int8),
        pl.col("non_home_discharge").fill_null(0).cast(pl.Int8),
    )
    pats = transform_sensitive_attributes(pats)

    if verbose:
        print(
            f"Subjects with in-hospital death: {pats.filter(pl.col('in_hosp_death') == 1).select('subject_id').n_unique()}, % of pts: {pats.filter(pl.col('in_hosp_death') == 1).select('subject_id').n_unique() / pats.select('subject_id').n_unique() * 100:.2f}"
        )
        print(
            f"Subjects with non-home discharge: {pats.filter(pl.col('non_home_discharge') == 1).select('subject_id').n_unique()}, % of pts: {pats.filter(pl.col('non_home_discharge') == 1).select('subject_id').n_unique() / pats.select('subject_id').n_unique() * 100:.2f}"
        )
    print("Collected patients table linked to ED attendances..")
    return pats.lazy() if use_lazy else pats


def read_icu_table(
    mimic4_ed_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the ICU stays table and join with admissions.

    Args:
        mimic4_ed_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: ICU stays table with joined admissions and derived columns.
    """
    icu = pl.read_csv(
        os.path.join(mimic4_ed_path, "icustays.csv.gz"),
        columns=["subject_id", "hadm_id", "intime", "outtime", "los"],
        dtypes=[pl.Int64, pl.Int64, pl.Datetime, pl.Datetime, pl.Float32],
        try_parse_dates=True,
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    if verbose:
        print(
            "Original number of ICU stays:",
            icu.shape[0],
            icu.select("subject_id").n_unique(),
        )
    icu = icu.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id"))
        & pl.col("hadm_id").is_in(admissions_data.select("hadm_id"))
    )
    if verbose:
        print(
            "Number of ICU stays with validated ED attendances:",
            icu.shape[0],
            icu.select("subject_id").n_unique(),
        )
    icu_eps = (
        admissions_data.join(icu, on=["subject_id", "hadm_id"], how="left")
        .sort(by=["subject_id", "hadm_id", "intime"])
        .unique(subset=["subject_id", "hadm_id"], keep="last")
    )

    icu_eps = icu_eps.with_columns(
        (
            (pl.col("intime") > pl.col("admittime"))
            & (pl.col("outtime") < pl.col("dischtime"))
        )
        .cast(pl.Int8)
        .alias("icu_admission"),
        pl.col("los").alias("icu_los_days"),
    )
    icu_eps = icu_eps.with_columns(
        pl.col("icu_admission").fill_null(0).cast(pl.Int8),
        pl.col("icu_los_days").fill_null(0).cast(pl.Int8),
    )
    icu_eps = icu_eps.drop(["los"])
    if verbose:
        print(
            f"Subjects with ICU admission: {icu_eps.filter(pl.col('icu_admission') == 1).select('subject_id').n_unique()}, % of pts: {icu_eps.filter(pl.col('icu_admission') == 1).select('subject_id').n_unique() / icu_eps.select('subject_id').n_unique() * 100:.2f}"
        )
    print("Collected ICU stay outcomes..")
    return icu_eps.lazy() if use_lazy else icu_eps


def read_d_icd_diagnoses_table(mimic4_path):
    """
    Read the ICD diagnoses dictionary table from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.

    Returns:
        pl.DataFrame: ICD diagnoses dictionary table.
    """
    d_icd = pl.read_csv(
        os.path.join(mimic4_path, "d_icd_diagnoses.csv.gz"),
        columns=["icd_code", "long_title"],
        dtypes=[pl.String, pl.String],
    )
    return d_icd


def read_diagnoses_table(
    mimic4_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    adm_last: pl.DataFrame | pl.LazyFrame,
    verbose: bool = True,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the diagnoses table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        adm_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up prior diagnoses.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Diagnoses table filtered and joined with admissions.
    """
    diag = pl.read_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz"),
        columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtypes=[pl.Int64, pl.Int64, pl.Int16, pl.String, pl.Int16],
    )
    diag_mapping = read_d_icd_diagnoses_table(mimic4_path)
    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(adm_last, pl.LazyFrame):
        adm_last = adm_last.collect()

    diag = diag.join(diag_mapping, on="icd_code", how="inner")
    if verbose:
        print("Original number of diagnoses:", len(diag))

    # Get list of eligible hospital episodes as historical data
    adm_lkup = admissions_data.join(
        adm_last.select(["subject_id", "edregtime"]).rename(
            {"edregtime": "last_edregtime"}
        ),
        on="subject_id",
        how="left",
    )
    adm_lkup = adm_lkup.filter(pl.col("edregtime") < pl.col("last_edregtime"))
    # Filter diagnoses for lookup episodes
    diag = diag.filter(pl.col("subject_id").is_in(adm_lkup.select("subject_id")))
    diag = diag.filter(pl.col("hadm_id").is_in(adm_lkup.select("hadm_id")))

    print("Collected diagnoses table..")
    return diag.lazy() if use_lazy else diag


def read_notes(
    admissions_data: pl.DataFrame | pl.LazyFrame,
    admits_last: pl.DataFrame | pl.LazyFrame,
    mimic4_path: str,
    verbose: bool = True,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess discharge summary and link Brief Hospital Course segments.

    Args:
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up notes history.
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Notes table joined with admissions and BHC segments.
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
        try_parse_dates=True,
    ).select(["subject_id", "hadm_id", "note_id", "charttime", "storetime", "text"])

    notes_ext = pl.read_csv(
        os.path.join(mimic4_path, "mimic-iv-bhc.csv"),
        dtypes=[pl.String, pl.String, pl.String, pl.Int64, pl.Int64],
    ).select(["note_id", "input", "target", "input_tokens", "target_tokens"])

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Merge with ED attendances cohort
    if verbose:
        print(
            "Original number of notes:",
            notes.shape[0],
            notes.select("subject_id").n_unique(),
        )
    notes = notes.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id").to_series())
    )
    if verbose:
        print(
            "Number of notes with validated ED attendances:",
            notes.shape[0],
            notes.select("subject_id").n_unique(),
        )
    notes = notes.join(notes_ext, how="left", on="note_id")
    if verbose:
        print(
            "Number of total matching preprocessed notes:",
            notes.filter(pl.col("input").is_not_null()).shape[0],
            notes.filter(pl.col("target").is_not_null()).shape[0],
        )
        print(
            "Unique patients with matching preprocessed notes:",
            notes.filter(pl.col("input").is_not_null()).select("subject_id").n_unique(),
            notes.filter(pl.col("target").is_not_null())
            .select("subject_id")
            .n_unique(),
        )

    adm_notes = (
        admissions_data.select(["subject_id", "hadm_id", "edregtime"])
        .join(
            notes.select(
                [
                    "note_id",
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "text",
                    "input",
                    "target",
                    "input_tokens",
                    "target_tokens",
                ]
            ),
            on=["subject_id", "hadm_id"],
            how="left",
        )
        .filter(pl.col("target").is_not_null())
    )

    ### Get previous hospital episodes as historical data
    adm_lkup = adm_notes.join(
        admits_last.select(["subject_id", "edregtime"]).rename(
            {"edregtime": "last_edregtime"}
        ),
        on="subject_id",
        how="left",
    ).filter(pl.col("edregtime") < pl.col("last_edregtime"))
    ## Replace more than one back-to-back '=' characters with ''
    adm_notes = adm_notes.with_columns(pl.col("target").str.replace_all(r"==+", ""))

    ### Get full notes history for each eligible patient
    adm_notes = adm_notes.filter(
        pl.col("hadm_id").is_in(adm_lkup.select("hadm_id").to_series())
    )

    return adm_notes.lazy() if use_lazy else adm_notes


def get_notes_population(
    adm_notes: pl.DataFrame | pl.LazyFrame,
    admit_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Get population of unique ED patients with existing note history.

    Args:
        adm_notes (pl.DataFrame | pl.LazyFrame): Notes table.
        admit_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for looking up notes history.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        tuple: Patients and Grouped notes table (ed_pts, notes_grouped) as DataFrames or LazyFrames.
    """
    if isinstance(adm_notes, pl.LazyFrame):
        adm_notes = adm_notes.collect(streaming=True)
    if isinstance(admit_last, pl.LazyFrame):
        admit_last = admit_last.collect()

    ### Aggregate historical notes data with demographics
    notes_grouped = adm_notes.groupby("subject_id").agg(
        [
            pl.col("hadm_id").n_unique().alias("num_summaries"),
            pl.col("input_tokens").sum().alias("num_input_tokens"),
            pl.col("target_tokens").sum().alias("num_target_tokens"),
            pl.col("target")
            .apply(lambda x: "<ENDNOTE> <STARTNOTE> ".join(x))
            .alias("target"),
        ]
    )
    ### Trim any leading or trailing whitespace
    notes_grouped = notes_grouped.with_columns(pl.col("target").str.strip_chars())
    ### Filter population with at least one note
    ed_pts = admit_last.filter(
        pl.col("subject_id").is_in(notes_grouped.select("subject_id"))
    )
    ## Save number of tokens per patient
    ed_pts = ed_pts.join(
        notes_grouped.select(
            ["subject_id", "num_summaries", "num_input_tokens", "num_target_tokens"]
        ),
        on="subject_id",
        how="left",
    )

    return (
        ed_pts.lazy() if use_lazy else ed_pts,
        notes_grouped.lazy() if use_lazy else notes_grouped,
    )


def read_omr_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the omr table (online medical records containing in-hospital measurements) from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_uom_map (dict, optional): Mapping for measurement units.

    Returns:
        pl.LazyFrame | pl.DataFrame: OMR table in long format.
    """
    vitalsign_uom_map = {
        "Temperature": "°F",
        "Heart rate": "bpm",
        "Respiratory rate": "insp/min",
        "Oxygen saturation": "%",
        "Systolic blood pressure": "mmHg",
        "Diastolic blood pressure": "mmHg",
        "BMI": "kg/m²",
    }
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    omr = pl.read_csv(
        os.path.join(mimic4_path, "omr.csv.gz"),
        dtypes=[pl.Int64, pl.String, pl.Int64, pl.String, pl.String],
    )
    omr = omr.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    omr = omr.with_columns(
        pl.col("chartdate").str.strptime(pl.Date, "%Y-%m-%d").alias("charttime")
    )
    # Drop seq_num and chartdate
    omr = omr.drop(["seq_num", "chartdate"])

    ### Prepare hospital measures time-series
    omr = omr.join(
        admits_last.select(["subject_id", "prev_edregtime", "prev_dischtime"]),
        on="subject_id",
        how="left",
    )
    omr = omr.filter(
        (pl.col("charttime") <= pl.col("prev_dischtime"))
        & (pl.col("charttime") >= pl.col("prev_edregtime"))
    )
    omr = omr.drop(["prev_edregtime", "prev_dischtime"])

    ### Requires reverting to pandas for string operations
    omr = omr.to_pandas()
    omr["result_name"] = np.where(
        omr["result_name"].str.contains("Blood Pressure"), "bp", omr["result_name"]
    )
    omr[["result_sysbp", "result_diabp"]] = omr["result_value"].str.split(
        "/", expand=True
    )
    omr["result_sysbp"] = pd.to_numeric(omr["result_sysbp"], errors="coerce")
    omr["result_diabp"] = pd.to_numeric(omr["result_diabp"], errors="coerce")
    omr["result_name"] = np.where(
        omr["result_name"].str.contains("BMI"), "bmi", omr["result_name"]
    )

    # Create separate rows for sysbp and diabp
    sysbp_measures = omr[["subject_id", "charttime", "result_sysbp"]].rename(
        columns={"result_sysbp": "value"}
    )
    sysbp_measures["label"] = "Systolic blood pressure"
    diabp_measures = omr[["subject_id", "charttime", "result_diabp"]].rename(
        columns={"result_diabp": "value"}
    )
    diabp_measures["label"] = "Diastolic blood pressure"

    # Concatenate the sysbp and diabp measures
    bp_measures = pd.concat([sysbp_measures, diabp_measures], axis=0)
    # Add BMI measurements
    bmi_measures = omr[omr["result_name"] == "bmi"][
        ["subject_id", "charttime", "result_value"]
    ].rename(columns={"result_value": "value"})
    bmi_measures["label"] = "BMI"
    omr = pd.concat([bp_measures, bmi_measures], axis=0)

    # Map the value_uom
    omr["valueuom"] = omr["label"].map(vitalsign_uom_map)
    omr["value"] = omr["value"].astype(np.float32)
    omr = pl.DataFrame(omr)

    return omr.lazy() if use_lazy else omr


def read_vitals_table(
    mimic4_ed_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_column_map: dict = None,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the vitalsign table from MIMIC-IV-ED.

    Args:
        mimic4_ed_path (str): Path to directory containing MIMIC-IV ED module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_column_map (dict, optional): Mapping for vital sign column names.
        vitalsign_uom_map (dict, optional): Mapping for vital sign units.

    Returns:
        pl.LazyFrame | pl.DataFrame: Vitals table in long format.
    """
    vitalsign_uom_map = {
        "Temperature": "°F",
        "Heart rate": "bpm",
        "Respiratory rate": "insp/min",
        "Oxygen saturation": "%",
        "Systolic blood pressure": "mmHg",
        "Diastolic blood pressure": "mmHg",
        "BMI": "kg/m²",
    }
    vitalsign_column_map = {
        "temperature": "Temperature",
        "heartrate": "Heart rate",
        "resprate": "Respiratory rate",
        "o2sat": "Oxygen saturation",
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
    }

    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    vitals = pl.read_csv(
        os.path.join(mimic4_ed_path, "vitalsign.csv.gz"),
        dtypes=[pl.Int64, pl.Int64, pl.Datetime, pl.String, pl.String, pl.String],
        try_parse_dates=True,
    )

    ### Prepare ed vital signs measures in long table format
    vitals = vitals.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    vitals = vitals.drop(["stay_id", "pain", "rhythm"])
    vitals = vitals.join(
        admits_last.select(["subject_id", "prev_edregtime", "prev_dischtime"]),
        on="subject_id",
        how="left",
    )
    vitals = vitals.filter(
        (pl.col("charttime") <= pl.col("prev_dischtime"))
        & (pl.col("charttime") >= pl.col("prev_edregtime"))
    )
    vitals = vitals.drop(["prev_edregtime", "prev_dischtime"])
    vitals = vitals.rename(vitalsign_column_map)
    vitals = vitals.melt(
        id_vars=["subject_id", "charttime"],
        value_vars=[
            "Temperature",
            "Heart rate",
            "Respiratory rate",
            "Oxygen saturation",
            "Systolic blood pressure",
            "Diastolic blood pressure",
        ],
        variable_name="label",
        value_name="value",
    ).sort(by=["subject_id", "charttime"])
    vitals = vitals.drop_nulls("value").with_columns(pl.col("value").cast(pl.Float64))
    vitals = vitals.with_columns(
        pl.col("label").map_dict(vitalsign_uom_map).alias("valueuom")
    )

    return vitals.lazy() if use_lazy else vitals


def read_labevents_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    include_items: str = "../config/lab_items.csv",
) -> pl.LazyFrame:
    """
    Read and preprocess the labevents table from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last admissions table for lookup.
        include_items (str): Path to file listing lab item IDs to include.

    Returns:
        pl.LazyFrame: Labevents table in long format.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    #  Load in csv using polars lazy API (requires table to be in csv format)
    labs_data = pl.scan_csv(
        os.path.join(mimic4_path, "labevents.csv"), try_parse_dates=True
    )
    d_items = (
        pl.read_csv(os.path.join(mimic4_path, "d_labitems.csv.gz"))
        .lazy()
        .select(["itemid", "label"])
    )
    # merge labitem id's with dict
    labs_data = labs_data.join(d_items, how="left", on="itemid")
    # select relevant columns
    labs_data = labs_data.select(
        ["subject_id", "charttime", "itemid", "label", "value", "valueuom"]
    ).with_columns(
        charttime=pl.col("charttime").cast(pl.Datetime), linksto=pl.lit("labevents")
    )
    # get eligible lab tests prior to current episode
    labs_data = labs_data.join(
        admits_last[["subject_id", "prev_edregtime", "prev_dischtime"]]
        .lazy()
        .with_columns(
            prev_edregtime=pl.col("prev_edregtime").cast(pl.Datetime),
            prev_dischtime=pl.col("prev_dischtime").cast(pl.Datetime),
        ),
        how="left",
        on="subject_id",
    )
    labs_data = labs_data.filter(
        (pl.col("charttime") <= pl.col("prev_dischtime"))
        & (pl.col("charttime") >= pl.col("prev_edregtime"))
    ).drop(["prev_edregtime", "prev_dischtime"])
    # get most common items (sample file contains top 50 itemids)
    if include_items is not None:
        # read txt file containing list of ids
        with open(include_items) as f:
            lab_items = list(f.read().splitlines())
        labs_data = labs_data.filter(
            pl.col("itemid").cast(pl.Utf8).is_in(set(lab_items))
        )

    labs_data = clean_labevents(labs_data)
    labs_data = labs_data.sort(by=["subject_id", "charttime"])

    return labs_data


def merge_events_table(
    vitals: pl.LazyFrame | pl.DataFrame,
    labs: pl.LazyFrame | pl.DataFrame,
    omr: pl.LazyFrame | pl.DataFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Merge vitals, labevents, and omr tables into a single events table for time-series modality.

    Args:
        vitals (pl.LazyFrame | pl.DataFrame): Vitals table.
        labs (pl.LazyFrame | pl.DataFrame): Labevents table.
        omr (pl.LazyFrame | pl.DataFrame): OMR table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: Merged events table.
    """
    ### Collect dataframes if lazy
    if isinstance(vitals, pl.LazyFrame):
        vitals = vitals.collect()
    if isinstance(labs, pl.LazyFrame):
        labs = labs.collect(streaming=True)
    if isinstance(omr, pl.LazyFrame):
        omr = omr.collect()

    # Combine vitals into a single table
    vitals = vitals.with_columns(pl.lit("vitals_measurements").alias("linksto"))
    vitals = vitals.with_columns(pl.col("value").cast(pl.Float64).drop_nulls())
    vitals = vitals.sort(by=["subject_id", "charttime"]).unique(
        subset=["subject_id", "charttime", "label"], keep="last"
    )
    vitals = vitals.with_columns(pl.lit(None).alias("itemid"))
    ### Reorder itemid columns to be third
    # vitals = vitals.to_pandas()
    vitals = vitals.select(
        ["subject_id", "charttime", "itemid", "label", "value", "valueuom", "linksto"]
    )
    # vitals = pl.DataFrame(vitals)
    vitals = vitals.with_columns(pl.col("charttime").cast(pl.String))
    events = labs.vstack(vitals)
    if verbose:
        print(
            f"# collected vitals records: {vitals.shape[0]} across {vitals.select('subject_id').n_unique()} patients."
        )
        print(
            f"# collected lab records: {labs.shape[0]} across {labs.select('subject_id').n_unique()} patients."
        )
    return events.lazy() if use_lazy else events


def get_population_with_measures(
    events: pl.DataFrame | pl.LazyFrame,
    admit_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Get population of unique ED patients with recorded measurements.

    Args:
        events (pl.DataFrame | pl.LazyFrame): Events table.
        admit_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for lookup.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Patient-level table with measurement counts.
    """
    if isinstance(events, pl.LazyFrame):
        events = events.collect(streaming=True)
    if isinstance(admit_last, pl.LazyFrame):
        admit_last = admit_last.collect()

    ### Aggregate historical notes data with demographics
    events_grouped = events.groupby("subject_id").agg(
        [pl.col("itemid").n_unique().alias("num_measures")]
    )
    ### Filter population with at least one measure
    ed_pts = admit_last.filter(
        pl.col("subject_id").is_in(events_grouped.select("subject_id"))
    )
    ## Save number of tokens per patient
    ed_pts = ed_pts.join(
        events_grouped.select(["subject_id", "num_measures"]),
        on="subject_id",
        how="left",
    )
    return ed_pts.lazy() if use_lazy else ed_pts


def read_medications_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    top_n: int = 50,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Get medication table from online administration record containing orders data.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        top_n (int): Includes the top N most commonly prescribed medications as count features.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table with medication features.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    meds = pl.read_csv(
        os.path.join(mimic4_path, "emar.csv.gz"),
        dtypes=[pl.Int64, pl.Datetime, pl.String, pl.String],
        columns=["subject_id", "charttime", "medication", "event_txt"],
        try_parse_dates=True,
    )
    ### Link relevant medications and prepare for parsing
    meds = meds.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    meds = meds.join(
        admits_last.select(["subject_id", "edregtime"]), on="subject_id", how="left"
    )
    meds = meds.with_columns(pl.col("edregtime").cast(pl.Datetime))
    meds = meds.filter(pl.col("charttime") < pl.col("edregtime"))
    meds = meds.drop_nulls(subset=["medication", "event_txt", "charttime"])
    ### Filter correctly administered medications
    meds = meds.filter(
        pl.col("event_txt").is_in(["Administered", "Confirmed", "Started"])
    )
    ### Generate drug-level features and append to EHR data
    admits_last = prepare_medication_features(
        meds, admits_last, top_n=top_n, use_lazy=use_lazy
    )
    return admits_last.lazy() if use_lazy else admits_last


def read_specialty_table(
    mimic4_path: str, admits_last: pl.DataFrame | pl.LazyFrame, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """
    Collect specialty-grouped count features from secondary care provider order history.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table with specialty features.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    poe = pl.read_csv(
        os.path.join(mimic4_path, "poe.csv.gz"),
        dtypes=[pl.Int64, pl.Datetime, pl.String],
        columns=["subject_id", "ordertime", "order_type"],
        try_parse_dates=True,
    )
    ### Link related orders
    poe = poe.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    poe = poe.sort(["subject_id", "ordertime"])
    poe = poe.join(
        admits_last.select(["subject_id", "edregtime"]), on="subject_id", how="left"
    )
    poe = poe.filter(pl.col("ordertime") < pl.col("edregtime"))
    ### Filter order types of interest (can be extended to capture specific treatments)
    poe = poe.filter(
        pl.col("order_type").is_in(
            [
                "Nutrition",
                "TPN",
                "Cardiology",
                "Radiology",
                "Neurology",
                "Respiratory",
                "Hemodialysis",
            ]
        )
    )
    poe_ids = poe.groupby(["subject_id", "order_type"]).agg(
        pl.count("order_type").alias("admin_proc_count")
    )
    poe_piv = poe_ids.pivot(
        values="admin_proc_count",
        index="subject_id",
        columns="order_type",
        aggregate_function="first",
    )
    ### Pivot table to create specialty count features
    poe_piv.columns = [rename_fields(col) for col in poe_piv.columns]
    poe_piv = poe_piv.with_columns(pl.col("*").fill_null(0))
    poe_piv_total = poe_ids.groupby("subject_id").agg(
        pl.count("order_type").alias("total_proc_count")
    )
    admits_last = admits_last.join(poe_piv_total, on="subject_id", how="left")
    admits_last = admits_last.join(poe_piv, on="subject_id", how="left")
    admits_last = admits_last.with_columns(
        pl.col("total_proc_count").fill_null(0).cast(pl.Int16)
    )
    ### Rename each specialty column with po_ prefix
    admits_last = admits_last.rename(
        {
            "Nutrition": "pon_nutrition",
            "TPN": "pon_tpn",
            "Cardiology": "pon_cardiology",
            "Radiology": "pon_radiology",
            "Neurology": "pon_neurology",
            "Respiratory": "pon_respiratory",
            "Hemodialysis": "pon_hemodialysis",
        }
    )
    ### Fill any NA values in the specialty columns with 0
    admits_last = admits_last.with_columns(
        [
            pl.col(col).fill_null(0)
            for col in admits_last.columns
            if col.startswith("pon_")
        ]
    )

    return admits_last.lazy() if use_lazy else admits_last


def save_multimodal_dataset(
    admits_last: pl.DataFrame | pl.LazyFrame,
    events: pl.DataFrame | pl.LazyFrame,
    notes: pl.DataFrame | pl.LazyFrame,
    use_events: bool = True,
    use_notes: bool = True,
    output_path: str = "../outputs/extracted_data",
):
    """
    Export datasets (EHR, events, notes) to CSV files for downstream processing.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Static EHR data.
        events (pl.DataFrame | pl.LazyFrame): Events time-series data.
        notes (pl.DataFrame | pl.LazyFrame): Notes data.
        use_events (bool): If True, save events data.
        use_notes (bool): If True, save notes data.
        output_path (str): Directory to save the output files.

    Returns:
        None
    """
    #### Save EHR data (required)
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()
    admits_last.to_pandas().to_csv(
        os.path.join(output_path, "ehr_static.csv"), index=False
    )
    #### Save time-series and notes modality data
    if use_events:
        if isinstance(events, pl.LazyFrame):
            events = events.collect(streaming=True)
        events.write_csv(os.path.join(output_path, "events_ts.csv"))
    if use_notes:
        if isinstance(notes, pl.LazyFrame):
            notes = notes.collect(streaming=True)
        notes.write_csv(os.path.join(output_path, "notes.csv"))
