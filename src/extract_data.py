import argparse  # noqa: I001
import os
import sys
import shutil
import gzip
import numpy as np
import polars as pl
from tqdm import tqdm

import utils.mimiciv as m4c
from utils.functions import get_n_unique_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from MIMIC-IV v2.2.")
    parser.add_argument(
        "mimic4_path", type=str, help="Directory containing downloaded MIMIC-IV data."
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
        help="Tables from which to read events. Can be any combination of: labevents, vitalsign",
        default=["vitalsign"],
    )
    parser.add_argument(
        "--labitems",
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
        help="Control verbosity. If true, will make more .collect() calls to compute dataset size.",
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
        "--thresh",
        type=int,
        default=2,
        help="Threshold for subject stratification (fractional days).",
    )

    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False (except for events tables - always uses lazymode).",
    )

    args, _ = parser.parse_known_args()
    mimic4_path = os.path.join(args.mimic4_path, "mimiciv", "2.2", "hosp")
    mimic4_ed_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "2.2", "ed")

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

    # Read in csv files
    patients = m4c.read_patients_table(mimic4_path, use_lazy=args.lazy)
    admits = m4c.read_admissions_table(mimic4_path, use_lazy=args.lazy)
    stays = m4c.read_stays_table(mimic4_ed_path, use_lazy=args.lazy)
    omr = m4c.read_omr_table(mimic4_path, use_lazy=args.lazy)

    # diagnoses = m4c.read_icd_diagnoses_table(mimic4_path, use_lazy=args.lazy)
    # ed_diagnoses = m4c.read_ed_icd_diagnoses_table(mimic4_ed_path, use_lazy=args.lazy)

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

    # remove any subjects with anchor_age < 18
    stays = m4c.filter_stays_on_age(stays)
    if args.verbose:
        print(
            f"REMOVE PATIENTS AGE < 18:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
        )

    stays = m4c.add_inhospital_mortality_to_stays(stays)
    # add height/weight data using omr table: https://mimic.mit.edu/docs/iv/modules/hosp/omr/
    # use tolerance to find closest value within a year of admission
    stays = m4c.add_omr_variable_to_stays(stays, omr, "height", tolerance=365)
    stays = m4c.add_omr_variable_to_stays(stays, omr, "weight", tolerance=365)

    ### IF LAZY COLLECT HERE SINCE .SAMPLE IS A DATAFRAME FUNCTION
    if type(stays) == pl.LazyFrame:
        print("Collecting...")
        stays = stays.collect(streaming=True)

    # filter by n subjects if specified (can be used to test/speed up processing)
    if args.sample is not None:
        # set the seed for reproducibility
        rng = np.random.default_rng(0)

        if not args.stratify:
            mode = "random"
            # random choice of n subjects (get all their stays)
            stays = stays.sample(n=args.sample, seed=0)

        else:
            mode = "stratified"
            # use los column to stratify such that the distribution of los >= 48 hours (2 days) is roughly balanced
            stays = stays.with_columns(
                los_flag=(pl.col("los") >= args.thresh).cast(pl.Boolean)
            )

            if args.stratify_level == "subject":
                # Get maximum n based on stratified samples
                max_stratified_n = min(
                    [
                        stays.filter(pl.col("los_flag"))
                        .unique(subset="subject_id")
                        .height,
                        stays.filter(~pl.col("los_flag"))
                        .unique(subset="subject_id")
                        .height,
                    ]
                )
                assert (
                    args.sample // 2 <= max_stratified_n
                ), f"Maximum number of subjects available per group is {max_stratified_n}. Choose a different value for args.sample"

                # negative subjects
                subject_ids = (
                    stays.filter(~pl.col("los_flag"))
                    .get_column("subject_id")
                    .sample(n=args.sample // 2, seed=0)
                    .to_list()
                )

                remaining_stays = stays.filter(
                    ~(pl.col("subject_id").is_in(subject_ids))
                )

                # positive subjects
                subject_ids += (
                    remaining_stays.filter(pl.col("los_flag"))
                    .get_column("subject_id")
                    .sample(n=args.sample // 2, seed=0)
                    .to_list()
                )

                stays = stays.filter(pl.col("subject_id").is_in(subject_ids)).drop(
                    "los_flag"
                )

            elif args.stratify_level == "stay":
                max_stratified_n = min(
                    [
                        stays.filter(pl.col("los_flag")).height,
                        stays.filter(~pl.col("los_flag")).height,
                    ]
                )

                assert (
                    args.sample // 2 <= max_stratified_n
                ), f"Maximum number of stays available per group is {max_stratified_n}. Choose a different value for args.sample"

                # alternative to stratify by stay instead of subject

                # negative stays
                negative_stays = stays.filter(~pl.col("los_flag")).sample(
                    n=args.sample // 2, seed=0
                )

                # positive stays
                positive_stays = stays.filter(pl.col("los_flag")).sample(
                    n=args.sample // 2, seed=0
                )

                stays = pl.concat([negative_stays, positive_stays])

        # stratify by length of stay
        if args.verbose:
            print(
                f"SELECTING {mode.upper()} SAMPLE OF {args.sample} {'SUBJECTS' if args.stratify_level == 'subject' else 'STAYS'}:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
            )

    # Write all stays
    stays.write_csv(os.path.join(args.output_path, "stays.csv"))

    # Break up into subject-level stays
    subjects = stays.unique(subset="subject_id").get_column("subject_id").to_list()
    # m4c.break_up_stays_by_subject(stays, args.output_path, subjects=subjects)

    items = (
        set(
            pl.read_csv(args.labitems)
            .unique(subset="itemid")
            .get_column("itemid")
            .cast(pl.UInt64)
            .to_numpy()
        )
        if args.labitems
        else [51221, 50912, 51301, 51265, 50971, 50983, 50931, 50893, 50960]
    )

    for table in tqdm(
        args.event_tables,
        desc="Processing events tables...",
        total=len(args.event_tables),
    ):
        mimic_dir = mimic4_path if table == "labevents" else mimic4_ed_path

        # read compressed and write to file since lazy polars API can only scan uncompressed csv's
        if not os.path.exists(os.path.join(mimic_dir, f"{table}.csv")):
            print(f"Uncompressing {table} data... (required)")
            with gzip.open(os.path.join(mimic_dir, f"{table}.csv.gz"), "rb") as f_in:
                with open(os.path.join(mimic_dir, f"{table}.csv"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        events = m4c.read_events_table(
            table,
            mimic_dir,
            include_items=items,
            include_subjects=subjects,
        )

        if table == "labevents":
            print(
                "Trying to impute missing hadm_ids using join: see https://mimic.mit.edu/docs/iv/modules/hosp/labevents/..."
            )
            # use admissions table to impute missing hadm_ids based on charttime
            events = m4c.get_hadm_id_from_admits(events, admits)

        if os.path.exists(os.path.join(args.output_path, "events.csv")):
            with open(os.path.join(args.output_path, "events.csv"), mode="ab") as f:
                events.write_csv(f, include_header=False)
        else:
            events.write_csv(os.path.join(args.output_path, "events.csv"))

    print("Data extracted.")
