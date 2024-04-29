# Code adapted from

import os

import icdmappings
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from . import util

ICD9 = 9


def read_patients_table(mimic4_path):
    pats = util.dataframe_from_csv(os.path.join(mimic4_path, "patients.csv.gz"))
    pats = pats[
        ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
    ]  # dod for out of hospital mortality
    pats.dod = pd.to_datetime(pats.dod)
    return pats


def read_omr_table(mimic4_path):
    omr = util.dataframe_from_csv(os.path.join(mimic4_path, "omr.csv.gz"))
    omr.chartdate = pd.to_datetime(omr.chartdate)
    return omr


def read_admissions_table(mimic4_path):
    admits = util.dataframe_from_csv(os.path.join(mimic4_path, "admissions.csv.gz"))
    admits = admits[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "insurance",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ]
    ]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    # add los column (in fractional days)
    admits["los"] = (admits.dischtime - admits.admittime) / pd.Timedelta(days=1)
    return admits


def read_stays_table(mimic4_ed_path):
    stays = util.dataframe_from_csv(os.path.join(mimic4_ed_path, "edstays.csv.gz"))
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays["los_ed"] = (stays.outtime - stays.intime) / pd.Timedelta(days=1)
    return stays


def read_icd_diagnoses_table(mimic4_path):
    codes = util.dataframe_from_csv(os.path.join(mimic4_path, "d_icd_diagnoses.csv.gz"))
    codes = codes[["icd_code", "icd_version", "long_title"]]
    diagnoses = util.dataframe_from_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz")
    )
    diagnoses = diagnoses.merge(
        codes,
        how="inner",
        left_on=["icd_code", "icd_version"],
        right_on=["icd_code", "icd_version"],
    )
    diagnoses[["subject_id", "hadm_id", "seq_num"]] = diagnoses[
        ["subject_id", "hadm_id", "seq_num"]
    ].astype(int)
    return diagnoses


def read_events_table_and_break_up_by_subject(
    table_path,
    table,
    output_path,
    items_to_keep=None,
    subjects_to_keep=None,
    mimic4_path=None,
):
    #  Load in csv using polars lazy API (requires RAM)
    print(f"Reading {table} table...")
    table_df = pl.read_csv(table_path).lazy()

    # add column for linksto
    table_df = table_df.with_columns(linksto=pl.lit(table))

    if "stay_id" not in table_df.columns:
        # add column for stay_id
        table_df = table_df.with_columns(stay_id=pl.lit(None))

    if "hadm_id" not in table_df.columns:
        # add column for stay_id
        table_df = table_df.with_columns(hadm_id=pl.lit(None))

    # if not specified then use all subjects in df
    if subjects_to_keep is not None:
        subjects_to_keep = (
            table_df.unique(subset="subject_id")
            .collect()
            .get_column("subject_id")
            .to_list()
        )

    # labevents only
    if table == "labevents":
        d_items = (
            pl.read_csv(os.path.join(mimic4_path, "d_labitems.csv.gz"))
            .lazy()
            .select(["itemid", "label"])
        )

        # merge labitem id's with dict
        table_df = table_df.join(d_items, on="itemid")

        if items_to_keep is not None:
            table_df = table_df.filter(pl.col("itemid").is_in(set(items_to_keep)))

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
        )

        # create empty itemid and manually add valueuom
        table_df = table_df.with_columns(itemid=pl.lit(None))
        table_df = table_df.with_columns(
            valueuom=pl.col("label").replace(vitalsign_uom_map)
        )

    # select relevant columns
    table_df = table_df.select(
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "itemid",
            "value",
            "valueuom",
            "label",
            "linksto",
        ]
    )

    # write events to subject-level directories
    # loop over subjects by filtering df, extracting all events/items and writing to events.csv
    for subject in tqdm(subjects_to_keep, desc=f"Processing {table} table"):
        events = table_df.filter(pl.col("subject_id") == subject).collect()

        subject_fp = os.path.join(output_path, str(subject), "events.csv")

        # write all events to a subject-level dir
        if not os.path.isdir(os.path.join(output_path, str(subject))):
            os.makedirs(os.path.join(output_path, str(subject)))

        if os.path.exists(subject_fp):
            # append to csv if already exists e.g., from another events table
            with open(subject_fp, "a") as output_file:
                output_file.write(events.write_csv(file=None, include_header=False))
        else:
            events.write_csv(file=subject_fp, include_header=True)


def read_ed_icd_diagnoses_table(mimic4_ed_path):
    codes = util.dataframe_from_csv(os.path.join(mimic4_ed_path, "diagnosis.csv.gz"))
    return codes


def convert_icd9_to_icd10(diagnoses_df, keep_original=True):
    print("Converting ICD9 codes to ICD10...".upper())
    mapper = icdmappings.Mapper()

    if keep_original:
        # keep original codes
        diagnoses_df = diagnoses_df.assign(icd_code_orig=diagnoses_df["icd_code"])
        diagnoses_df = diagnoses_df.assign(icd_version_orig=diagnoses_df["icd_version"])

    # update 'icd_code' and 'icd_version' to ICD10
    idx = diagnoses_df["icd_version"] == ICD9
    diagnoses_df.loc[idx, "icd_code"] = mapper.map(
        diagnoses_df.loc[idx, "icd_code"], source="icd9", target="icd10"
    )

    diagnoses_df = diagnoses_df.assign(icd_version=10)
    return diagnoses_df


def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[["icd_code", "icd_version", "long_title"]]

    codes = codes.drop_duplicates().set_index("icd_code")

    codes["count"] = diagnoses.groupby("icd_code")["stay_id"].count()
    codes["count"] = codes["count"].fillna(0).astype(int)
    codes = codes[codes["count"] > 0]
    if output_path:
        codes.to_csv(output_path, index_label="icd_code")
    return codes.sort_values("count", ascending=False).reset_index()


def remove_stays_without_admission(stays):
    stays = stays[stays.disposition == "ADMITTED"].dropna(subset=["hadm_id"])
    return stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]


def merge_on_subject(table1, table2):
    return table1.merge(
        table2, how="inner", left_on=["subject_id"], right_on=["subject_id"]
    )


def merge_on_subject_admission(table1, table2):
    return table1.merge(
        table2,
        how="inner",
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "hadm_id"],
    )


def merge_on_subject_stay_admission(table1, table2, suffixes=("_x", "_y")):
    return table1.merge(
        table2,
        how="inner",
        left_on=["subject_id", "hadm_id", "stay_id"],
        right_on=["subject_id", "hadm_id", "stay_id"],
        suffixes=suffixes,
    )


# unlikely that age will change by more than 3 years between ED and admission to hospital so not using this function

# def add_age_to_stays(stays):
#     stays["age"] = stays.apply(
#         lambda e: (
#             (e["admittime"].to_pydatetime().year - e["anchor_year"]) // 3
#             + e["anchor_age"]
#         ),
#         axis=1,
#     )  # increases age every 3 years based on time elapsed between admission year and birth year
#     stays.loc[stays.age < 0, "age"] = 91  # set negatives as max age
#     return stays


def add_omr_variable_to_stays(stays, mimic4_path, variable, tolerance=None):
    # get value of variable on stay/admission date using omr record's date
    # use tolerance to allow elapsed time between dates
    omr = read_omr_table(mimic4_path)
    omr_names = [x for x in omr.result_name.unique() if variable.lower() in x.lower()]

    def get_values_from_omr(omr, table, result_names, time_col: str = None):
        table["admitdate"] = pd.to_datetime(table["admittime"].dt.date)
        omr = omr.merge(table[["subject_id", "admitdate"]], on="subject_id")
        omr["diff"] = np.abs(
            (omr["admitdate"] - omr["chartdate"]) / pd.Timedelta(days=1)
        )

        if tolerance is None:
            match = omr[omr["diff"] == 0].sort_values(by=["diff", "seq_num"])
        else:
            # allow window (days) of tolerance
            match = omr[omr["diff"] < tolerance].sort_values(by=["diff", "seq_num"])

        values = match[match["result_name"].isin(result_names)].result_value.values

        if len(values) != 0:
            return values[
                0
            ]  # take first value since sorted by diff (so will be the closest), then seq_num
        else:
            return np.nan

    stays[variable.lower()] = get_values_from_omr(
        omr, stays, omr_names, time_col="admittime"
    )

    return stays


def add_inhospital_mortality_to_stays(stays):
    if "hospital_expire_flag" in stays:
        stays = stays.rename(columns={"hospital_expire_flag": "mortality"})
    else:
        mortality = stays.dod.notnull() & (
            (stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)
        )
        mortality = mortality | (
            stays.deathtime.notnull()
            & (
                (stays.admittime <= stays.deathtime)
                & (stays.dischtime >= stays.deathtime)
            )
        )
        stays["mortality"] = mortality.astype(int)
    return stays


# not used, if died in ED not considered
def add_ined_mortality_to_stays(stays):
    mortality = stays.dod.notnull() & (
        (stays.intime <= stays.dod) & (stays.outtime >= stays.dod)
    )
    mortality = mortality | (
        stays.deathtime.notnull()
        & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime))
    )
    stays["mortality_ined"] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_stays(stays, min_nb_stays=1, max_nb_stays=1):
    # Only keep hospital admissions that are associated with a certain number of ED stays (within min and max)
    to_keep = stays.groupby("hadm_id").count()[["stay_id"]].reset_index()
    to_keep = to_keep[
        (to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)
    ][["hadm_id"]]
    stays = stays.merge(to_keep, how="inner", left_on="hadm_id", right_on="hadm_id")
    return stays


def filter_stays_on_age(stays, min_age=18, max_age=np.inf):
    # must have already added age to stays table
    stays = stays[(stays.anchor_age >= min_age) & (stays.anchor_age <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays, by_col="stay_id"):
    return diagnoses.merge(
        stays[["subject_id", "hadm_id", "stay_id"]].drop_duplicates(),
        how="inner",
        left_on=["subject_id", by_col],
        right_on=["subject_id", by_col],
    )


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up stays by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        stays[stays.subject_id == subject_id].sort_values(by="intime").to_csv(
            os.path.join(dn, "stays.csv"), index=False
        )


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up diagnoses by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        diagnoses[diagnoses.subject_id == subject_id].sort_values(
            by=["stay_id", "seq_num"]
        ).to_csv(os.path.join(dn, "diagnoses.csv"), index=False)
