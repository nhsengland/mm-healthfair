import os

import icdmappings
import numpy as np
import polars as pl
from tqdm import tqdm

ICD9 = 9


def read_patients_table(mimic4_path, lazy_mode=False):
    pats = pl.read_csv(
        os.path.join(mimic4_path, "patients.csv.gz"),
        columns=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        dtypes=[pl.UInt64, pl.String, pl.UInt64, pl.UInt64, pl.Datetime],
    )
    return pats.lazy() if lazy_mode else pats


def read_omr_table(mimic4_path, lazy_mode=False):
    omr = pl.read_csv(
        os.path.join(mimic4_path, "omr.csv.gz"),
        dtypes=[pl.UInt64, pl.Datetime, pl.UInt64, pl.String, pl.String],
    )
    return omr.lazy() if lazy_mode else omr


def read_admissions_table(mimic4_path, lazy_mode=False):
    admits = pl.read_csv(
        os.path.join(mimic4_path, "admissions.csv.gz"),
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "insurance",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
        dtypes=[
            pl.UInt64,
            pl.UInt64,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.String,
            pl.String,
            pl.String,
            pl.UInt64,
        ],
    )
    admits = admits.with_columns(
        ((pl.col("dischtime") - pl.col("admittime")) / pl.duration(days=1)).alias("los")
    )
    return admits.lazy() if lazy_mode else admits


def read_stays_table(mimic4_ed_path, lazy_mode=False):
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
        dtypes=[pl.UInt64, pl.UInt64, pl.UInt64, pl.Datetime, pl.Datetime, pl.String],
    )
    # stays = stays.with_columns(((pl.col("outtime") - pl.col("intime"))/pl.duration(days=1)).alias("los_ed"))
    return stays.lazy() if lazy_mode else stays


def read_icd_diagnoses_table(mimic4_path, lazy_mode=False):
    codes = pl.read_csv(os.path.join(mimic4_path, "d_icd_diagnoses.csv.gz"))
    diagnoses = pl.read_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz"),
        dtypes=[pl.UInt64, pl.UInt64, pl.UInt64, pl.String, pl.String],
    )
    diagnoses = diagnoses.merge(
        codes,
        how="inner",
        on=["icd_code", "icd_version"],
    )
    return diagnoses.lazy() if lazy_mode else diagnoses


def read_ed_icd_diagnoses_table(mimic4_ed_path, lazy_mode=False):
    diagnoses = pl.read_csv(
        os.path.join(mimic4_ed_path, "diagnosis.csv.gz"),
        dtypes=[pl.UInt64, pl.UInt64, pl.UInt64, pl.String, pl.UInt64, pl.String],
    ).lazy()
    return diagnoses.lazy() if lazy_mode else diagnoses


def read_events_table_and_break_up_by_subject(
    table_path,
    table,
    output_path,
    items_to_keep=None,
    subjects_to_keep=None,
    mimic4_path=None,
):
    #  Load in csv using polars lazy API (requires table to be in csv format)
    print(f"Reading {table} table...")
    table_df = pl.scan_csv(table_path)

    # add column for linksto
    table_df = table_df.with_columns(linksto=pl.lit(table))

    if "stay_id" not in table_df.columns:
        # add column for stay_id
        table_df = table_df.with_columns(stay_id=pl.lit(None, dtype=pl.UInt64))

    if "hadm_id" not in table_df.columns:
        # add column for stay_id
        table_df = table_df.with_columns(hadm_id=pl.lit(None, dtype=pl.UInt64))

    # if not specified then use all subjects in df
    # if subjects_to_keep is not None:
    #     subjects_to_keep = (
    #         table_df.unique(subset="subject_id")
    #         .collect()
    #         .get_column("subject_id")
    #         .to_list()
    #     )

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
        ).sort(by="charttime")

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
    ).collect(streaming=True)

    # write events to subject-level directories
    # loop over subjects by filtering df, extracting all events/items and writing to events.csv
    for subject in tqdm(subjects_to_keep, desc=f"Processing {table} table"):
        events = table_df.filter(pl.col("subject_id") == subject)

        subject_fp = os.path.join(output_path, str(subject), "events.csv")

        # write all events to a subject-level dir
        if not os.path.isdir(os.path.join(output_path, str(subject))):
            os.makedirs(os.path.join(output_path, str(subject)))

        if os.path.exists(subject_fp):
            # append to csv if already exists e.g., from another events table
            with open(subject_fp, "a") as output_file:
                output_file.write(events.write_csv(include_header=False))
        else:
            events.write_csv(file=subject_fp, include_header=True)


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
    return (
        stays.filter(pl.col("disposition") == "ADMITTED")
        .drop_nulls(subset="hadm_id")
        .drop("disposition")
    )


def merge_on_subject(table1, table2):
    return table1.join(table2, how="inner", on="subject_id")


def merge_on_subject_admission(table1, table2):
    return table1.join(table2, how="inner", on=["subject_id", "hadm_id"])


def merge_on_subject_stay_admission(table1, table2, suffixes=("_x", "_y")):
    return table1.join(
        table2,
        how="inner",
        on=["subject_id", "hadm_id", "stay_id"],
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
    omr = read_omr_table(mimic4_path).drop("seq_num")
    omr_names = [
        x
        for x in omr.unique(subset="result_name").get_column("result_name").to_list()
        if variable.lower() in x.lower()
    ]

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


def add_inhospital_mortality_to_stays(stays):
    if "hospital_expire_flag" in stays.columns:
        stays = stays.rename({"hospital_expire_flag": "mortality"})
    else:
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


# not used, if died in ED not considered
def add_ined_mortality_to_stays(stays):
    stays = stays.with_columns(
        (
            pl.col("dod").is_not_null()
            & (
                (pl.col("inttime") <= pl.col("dod"))
                & (pl.col("outtime") >= pl.col("dod"))
            )
        )
        | (
            pl.col("deathtime").is_not_null()
            & (
                (pl.col("inttime") <= pl.col("deathtime"))
                & (pl.col("outtime") >= pl.col("deathtime"))
            )
        )
        .cast(pl.UInt8)
        .alias("mortality_ined")
    )

    return stays


def filter_admissions_on_nb_stays(stays, min_nb_stays=1, max_nb_stays=1):
    # Only keep hospital admissions that are associated with a certain number of ED stays (within min and max)
    to_keep = stays.group_by("hadm_id").agg(pl.col("stay_id").count())
    to_keep = to_keep.filter(
        (pl.col("stay_id") >= min_nb_stays) & (pl.col("stay_id") <= max_nb_stays)
    ).select("hadm_id")
    stays = stays.join(to_keep, how="inner", on="hadm_id")
    return stays


def filter_stays_on_age(stays, min_age=18, max_age=np.inf):
    # must have already added age to stays table
    stays = stays.filter(
        (pl.col("anchor_age") >= min_age) & (pl.col("anchor_age") <= max_age)
    )
    return stays


def filter_diagnoses_on_stays(diagnoses, stays, by_col="stay_id"):
    return diagnoses.join(
        stays.select(["subject_id", "hadm_id", "stay_id"]).unique(),
        how="inner",
        on=["subject_id", by_col],
    )


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = (
        stays.unique(subset="subject_id").get_column("subject_id").to_list()
        if subjects is None
        else subjects
    )
    nb_subjects = len(subjects)
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up stays by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        stays.filter(pl.col("subject_id") == subject_id).sort(by="intime").write_csv(
            os.path.join(dn, "stays.csv")
        )


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = (
        diagnoses.unique(subset="subject_id").get_column("subject_id").to_list()
        if subjects is None
        else subjects
    )
    nb_subjects = len(subjects)
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up diagnoses by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except Exception:
            pass

        diagnoses.filter(pl.col("subject_id" == subject_id)).sort(
            by=["stay_id", "seq_num"]
        ).write_csv(os.path.join(dn, "diagnoses.csv"))
