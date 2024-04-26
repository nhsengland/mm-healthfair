# Code adapted from

import csv
import os

import icdmappings
import numpy as np
import pandas as pd
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


def map_itemids_to_labels(events, item_dict):
    return events.merge(item_dict[["itemid", "label"]], on="itemid")


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


def read_labevents(table, mimic4_path):
    return map_itemids_to_labels(
        table, util.dataframe_from_csv(os.path.join(mimic4_path, "d_labitems.csv.gz"))
    )


def map_itemid_to_label(itemid, label_dict):
    return label_dict[label_dict["itemid"] == itemid].label.values[0]


def read_vitalsign(table):
    vitalsign_column_map = {
        "temperature": "Temperature",
        "heartrate": "Heart rate",
        "resprate": "Respiratory rate",
        "o2sat": "Oxygen saturation",
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
    }

    table = table.rename(vitalsign_column_map, axis=1)
    table = util.melt_table(
        table,
        id_vars=["subject_id", "stay_id", "charttime"],
        value_vars=[
            "Temperature",
            "Heart rate",
            "Respiratory rate",
            "Oxygen saturation",
            "Systolic blood pressure",
            "Diastolic blood pressure",
        ],
    )

    table = table.rename({"variable": "label"}, axis=1)

    # create empty itemid and manually add valueuom
    table["itemid"] = None

    vitalsign_uom_map = {
        "Temperature": "°F",
        "Heart rate": "bpm",
        "Respiratory rate": "insp/min",
        "Oxygen saturation": "%",
        "Systolic blood pressure": "mmHg",
        "Diastolic blood pressure": "mmHg",
    }
    table["valueuom"] = table["label"].map(vitalsign_uom_map)

    return table


def read_events_table_by_row(table, mimic4_path=None):
    if "labevent_id" in table.columns:
        table = read_labevents(table, mimic4_path)
    else:
        table = read_vitalsign(table)

    for _, row in table.iterrows():
        yield row


def read_events_table_and_break_up_by_subject(
    table_path,
    table,
    output_path,
    items_to_keep=None,
    subjects_to_keep=None,
    mimic4_path=None,
    impute_missing_hadm_id=True,
    chunksize=None,
):
    obs_header = [
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

    class DataStats:
        def __init__(self):
            self.curr_subject_id = ""
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass
        fn = os.path.join(dn, "events.csv")
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, "w")
            f.write(",".join(obs_header) + "\n")
            f.close()
        w = csv.DictWriter(
            open(fn, "a"), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL
        )
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    if table == "labevents":
        if items_to_keep is not None:
            # set specific itemids that we are interested in (labevents only)
            items_to_keep = set([s for s in items_to_keep])

    if subjects_to_keep is not None:
        subjects_to_keep = set([s for s in subjects_to_keep])

    print(f"Calculating total size of {table} table...")
    total_num_rows = util.count_rows(table_path)

    # Read/write events in chunks
    for chunk in tqdm(
        util.dataframe_from_csv(table_path, chunksize=chunksize),
        total=(total_num_rows // chunksize),
        desc=f"Processing {table} table",
    ):
        for row in read_events_table_by_row(chunk, mimic4_path=mimic4_path):
            if (subjects_to_keep is not None) and (
                row["subject_id"] not in subjects_to_keep
            ):
                continue

            if (items_to_keep is not None) and (row["itemid"] not in items_to_keep):
                continue

            if table == "labevents" and impute_missing_hadm_id:
                # impute missing hadm_ids from admissions data

                admits = util.dataframe_from_csv(
                    os.path.join(mimic4_path, "admissions.csv.gz"),
                    usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                )

                def get_hadm_id_from_charttime(time, subject_id, admits):
                    # create bool
                    idx = (
                        (admits["admittime"] <= time)
                        & (time <= admits["dischtime"])
                        & (admits["subject_id"] == subject_id)
                    )

                    if any(idx) is True:
                        # apply bool and return value
                        return admits[idx].hadm_id.values[0]
                    else:
                        return np.nan

                row["hadm_id"] = (
                    get_hadm_id_from_charttime(row.charttime, row.subject_id, admits)
                    if np.isnan(row["hadm_id"])
                    else row["hadm_id"]
                )

            row_out = {
                "subject_id": row["subject_id"],
                # labevents stored some hadm_id's as floats so converts, and if missing then record as ''
                "hadm_id": ""
                if ("hadm_id" not in row) or np.isnan(row["hadm_id"])
                else int(row["hadm_id"]),
                "stay_id": "" if "stay_id" not in row else row["stay_id"],
                "charttime": row["charttime"],
                "itemid": row["itemid"],
                "value": row["value"],
                "valueuom": row["valueuom"],
                "label": row["label"],
                "linksto": table,
            }

            if data_stats.curr_subject_id not in ("", row["subject_id"]):
                write_current_observations()
            data_stats.curr_obs.append(row_out)
            data_stats.curr_subject_id = row["subject_id"]

        if data_stats.curr_subject_id != "":
            write_current_observations()
