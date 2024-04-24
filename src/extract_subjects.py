import argparse  # noqa: I001
import os
import sys
import shutil

import numpy as np

import data.mimiciv as m4c
from data.util import dataframe_from_csv

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
    default=["labevents"],
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
    "--quiet",
    "-q",
    dest="verbose",
    action="store_false",
    help="Suspend printing of details",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="TEST MODE: process only 1000 subjects, 1000000 events.",
)
args, _ = parser.parse_known_args()

if os.path.exists(args.output_path):
    response = input("Will need to overwriting existing directory... continue? (y/n)")
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
        f"START:\n\tED STAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}"
    )

stays = m4c.remove_stays_without_admission(stays)
if args.verbose:
    print(
        f"REMOVE ED STAYS WITHOUT ADMISSION:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}"
    )

stays = m4c.merge_on_subject_admission(
    admits, stays
)  # order matters here ad stays has hadm_id as floats so merge with hadm_id instead
stays = m4c.merge_on_subject(stays, patients)

# ensures one stay per hospital admission
stays = m4c.filter_admissions_on_nb_stays(stays)
if args.verbose:
    print(
        f"REMOVE MULTIPLE ED STAYS PER ADMIT:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}"
    )

stays = m4c.add_age_to_stays(stays)  # ! not used, could remove
stays = m4c.add_omr_variable_to_stays(stays, args.mimic4_path, "height", tolerance=365)
stays = m4c.add_omr_variable_to_stays(
    stays, args.mimic4_path, "weight", tolerance=365
)  # using tolerance to find value

stays = m4c.add_inhospital_mortality_to_stays(
    stays
)  # will use hospital_expire_flag if exists
stays = m4c.filter_stays_on_age(stays)
if args.verbose:
    print(
        f"REMOVE PATIENTS AGE < 18:\n\tSTAY_IDs: {stays.stay_id.unique().shape[0]}\n\tHADM_IDs: {stays.hadm_id.unique().shape[0]}\n\tSUBJECT_IDs: {stays.subject_id.unique().shape[0]}"
    )

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

if args.test:  # test rest of script with 100 subjects
    pat_idx = np.random.choice(patients.shape[0], size=100)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(
        patients[["subject_id"]], left_on="subject_id", right_on="subject_id"
    )
    print("Using only", stays.shape[0], "stays")

subjects = stays.subject_id.unique()
m4c.break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
m4c.break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)
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
    try:
        if os.path.exists(os.path.join(args.mimic4_path, f"{table}.csv.gz")):
            table_path = os.path.join(args.mimic4_path, f"{table}.csv.gz")
        elif os.path.exists(os.path.join(args.mimic4_ed_path, f"{table}.csv.gz")):
            table_path = os.path.join(args.mimic4_ed_path, f"{table}.csv.gz")
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
