import argparse  # noqa: I001
import os
import sys
import shutil
import gzip
import numpy as np

import data.mimiciv as m4c
from data.util import dataframe_from_csv, get_n_unique_values

parser = argparse.ArgumentParser(
    description="Extract per-subject data from MIMIC-IV CSV files."
)
parser.add_argument(
    "mimic4_path", type=str, help="Directory containing MIMIC-IV HOSP CSV files."
)
parser.add_argument(
    "--mimic4_ed_path",
    "-ed",
    type=str,
    help="Directory containing MIMIC-IV ED CSV files.",
    required=True,
)
parser.add_argument(
    "--output_path",
    "-o",
    type=str,
    help="Directory where per-subject data should be written.",
    required=True,
)
parser.add_argument(
    "--event_tables",
    "-e",
    type=str,
    nargs="+",
    help="Tables from which to read events. Can be any from: labevents, vitalsign",
    default=["vitalsign"],
)
parser.add_argument(
    "--keep_items",
    "-i",
    type=str,
    help="CSV containing list of ITEMIDs to keep from labevents.",
)
parser.add_argument(
    "--verbose",
    "-v",
    dest="verbose",
    action="store_true",
    default=True,
    help="Verbosity in output",
)
parser.add_argument(
    "--sample",
    "-s",
    type=int,
    help="Number of subjects/stays to include.",
)
parser.add_argument(
    "--stratify",
    action="store_true",
    help="Whether to randomly stratify subjects by los.",
)
parser.add_argument(
    "--stratify_level",
    type=str,
    default="stay",
    help="Whether to stratify by subjects or stay.",
)
parser.add_argument(
    "--los_thresh",
    type=int,
    default=2,
    help="Threshold for subject stratification (fractional days).",
)
parser.add_argument(
    "--chunksize",
    "-cs",
    type=int,
    help="Chunksize to process large amount of data. Defaults to None (no chunking).",
)

args, _ = parser.parse_known_args()

if os.path.exists(args.output_path):
    response = input("Will need to overwrite existing directory... continue? (y/n)")
    if response == "y":
        try:
            shutil.rmtree(args.output_path)  # delete old dir
            os.makedirs(args.output_path)  # make new dir
        except OSError as ex:
            print(ex)
            sys.exit()
    else:
        print("Exiting..")
        sys.exit()
else:
    print(f"Creating output directory for extracted subjects at {args.output_path}")
    os.makedirs(args.output_path)

patients = m4c.read_patients_table(args.mimic4_path)
admits = m4c.read_admissions_table(args.mimic4_path)
stays = m4c.read_stays_table(args.mimic4_ed_path)

if args.verbose:
    print(
        f"START:\n\tED STAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
    )

stays = m4c.remove_stays_without_admission(stays)
if args.verbose:
    print(
        f"REMOVE ED STAYS WITHOUT ADMISSION:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
    )

# order matters here ad stays has hadm_id as floats so merge with hadm_id instead
stays = m4c.merge_on_subject_admission(admits, stays)

# add patient info to admissions data
stays = m4c.merge_on_subject(stays, patients)

# ensure one ed stay per hospital admission
stays = m4c.filter_admissions_on_nb_stays(stays)
if args.verbose:
    print(
        f"REMOVE MULTIPLE ED STAYS PER ADMIT:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
    )

# does nothing if "hospital_expire_flag" is available
stays = m4c.add_inhospital_mortality_to_stays(stays)

# remove any subjects with anchor_age < 18
stays = m4c.filter_stays_on_age(stays)
if args.verbose:
    print(
        f"REMOVE PATIENTS AGE < 18:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
    )

# filter by n subjects if specified (can be used to test/speed up processing)
if args.sample is not None:
    # set the seed for reproducibility
    rng = np.random.default_rng(0)

    if not args.stratify:
        mode = "random"
        # random choice of n subjects (get all their stays)
        subject_ids = rng.choice(
            stays.subject_id.unique(), size=args.sample, replace=False
        )

    else:
        mode = "stratified"
        # use los column to stratify such that the distribution of los >= 48 hours (2 days) is roughly balanced
        stays["los_flag"] = (stays["los"] >= args.los_thresh).astype(bool)

        if args.stratify_level == "subject":
            max_stratified_n = min(
                [
                    len(stays[~stays["los_flag"]].subject_id.unique()),
                    len(stays[stays["los_flag"]].subject_id.unique()),
                ]
            )
            assert (
                args.sample // 2 <= max_stratified_n
            ), f"Maximum number of subjects available per group is {max_stratified_n}. Choose a different value for args.sample"

            # negative subjects
            subject_ids = list(
                rng.choice(
                    stays[~stays["los_flag"]].subject_id.unique(),
                    size=args.sample // 2,
                    replace=False,
                    random_state=0,
                )
            )
            remaining_stays = stays.query("subject_id not in @subject_ids")
            # positive subjects
            subject_ids += list(
                rng.choice(
                    remaining_stays[remaining_stays["los_flag"]].subject_id.unique(),
                    size=args.sample // 2,
                    replace=False,
                )
            )
            stays = stays.query("subject_id in @subject_ids")

        elif args.stratify_level == "stay":
            max_stratified_n = min(
                [
                    len(stays[~stays["los_flag"]]),
                    len(stays[stays["los_flag"]]),
                ]
            )
            assert (
                args.sample // 2 <= max_stratified_n
            ), f"Maximum number of stays available per group is {max_stratified_n}. Choose a different value for args.sample"

            # alternative to stratify by stay instead of subject

            # negative stays
            stay_ids = list(
                rng.choice(
                    stays[~stays["los_flag"]].stay_id,
                    size=args.sample // 2,
                    replace=False,
                )
            )

            # positive stays
            stay_ids += list(
                rng.choice(
                    stays[stays["los_flag"]].stay_id,
                    size=args.sample // 2,
                    replace=False,
                )
            )
            stays = stays.query("stay_id in @stay_ids")

    # stratify by length of stay
    if args.verbose:
        print(
            f"SELECTING {mode.upper()} SAMPLE OF {args.sample} {'SUBJECTS' if args.stratify_level == 'subject' else 'STAYS'}:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}"
        )

# add height/weight data using omr table: https://mimic.mit.edu/docs/iv/modules/hosp/omr/
# use tolerance to find closest value within a year of admission
stays = m4c.add_omr_variable_to_stays(stays, args.mimic4_path, "height", tolerance=365)
stays = m4c.add_omr_variable_to_stays(stays, args.mimic4_path, "weight", tolerance=365)
stays.to_csv(os.path.join(args.output_path, "all_stays.csv"), index=False)

diagnoses = m4c.read_icd_diagnoses_table(args.mimic4_path)
diagnoses = m4c.convert_icd9_to_icd10(
    diagnoses, keep_original=False
)  # convert to ICD10
diagnoses = m4c.filter_diagnoses_on_stays(
    diagnoses, stays, by_col="hadm_id"
)  # use hadm_id to filter diagnoses by selected stays

ed_diagnoses = m4c.read_ed_icd_diagnoses_table(args.mimic4_ed_path)
ed_diagnoses = m4c.convert_icd9_to_icd10(
    ed_diagnoses, keep_original=False
)  # convert to ICD10
ed_diagnoses = m4c.filter_diagnoses_on_stays(
    ed_diagnoses, stays
)  # use stay_id (default) to filter diagnoses by selected stays

# merge into one diagnoses table on both hadm_id and stay_id
diagnoses = m4c.merge_on_subject_stay_admission(
    diagnoses, ed_diagnoses, suffixes=(None, "_ed")
)

diagnoses.to_csv(os.path.join(args.output_path, "all_diagnoses.csv"), index=False)
m4c.count_icd_codes(
    diagnoses, output_path=os.path.join(args.output_path, "diagnosis_counts.csv")
)

subjects = stays.subject_id.unique()
m4c.break_up_stays_by_subject(stays, args.output_path, subjects=subjects)

# for now skip this line since we aren't interested in the subject-level diagnoses
# m4c.break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)

items_to_keep = (
    set(
        [
            int(itemid)
            for itemid in dataframe_from_csv(args.keep_items)["itemid"].unique()
        ]
    )
    if args.keep_items
    else None
)
for table in args.event_tables:
    mimic_dir = args.mimic4_path if table == "labevents" else args.mimic4_ed_path
    try:
        if os.path.exists(os.path.join(mimic_dir, f"{table}.csv")):
            table_path = os.path.join(mimic_dir, f"{table}.csv")
        # read compressed and write to file since lazy polars API can only scan uncompressed csv's
        elif os.path.exists(os.path.join(mimic_dir, f"{table}.csv.gz")):
            print(f"Uncompressing {table} data... (required)")
            with gzip.open(os.path.join(mimic_dir, f"{table}.csv.gz"), "rb") as f_in:
                with open(os.path.join(mimic_dir, f"{table}.csv"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            table_path = os.path.join(mimic_dir, f"{table}.csv")

    except Exception:
        print(f"Event tables for {table} cannot be found in MIMICIV directory.")

    m4c.read_events_table_and_break_up_by_subject(
        table_path,
        table,
        args.output_path,
        items_to_keep=items_to_keep,
        subjects_to_keep=subjects,
        mimic4_path=args.mimic4_path,
    )
print("Subjects extracted.")
