import argparse  # noqa: I001
import os
import sys
import shutil
import gzip
import numpy as np
import polars as pl
from tqdm import tqdm

import utils.mimiciv as m4c
from utils.functions import get_n_unique_values, impute_from_df, read_from_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from MIMIC-IV v3.1.")
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
        default=["labevents", "vitalsign"],
    )

    parser.add_argument("--include_notes", "-n", action="store_true")

    parser.add_argument(
        "--labitems",
        "-i",
        type=str,
        help="Text file containing list of ITEMIDs to use from labevents.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity. If true, will make more .collect() calls to compute dataset size.",
    )
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        help="Number of stays to sample (random).",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False (except for events tables - always uses lazymode).",
    )

    args, _ = parser.parse_known_args()
    mimic4_path = os.path.join(args.mimic4_path, "mimiciv", "3.1", "hosp")
    mimic4_ed_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "3.1", "ed")
    mimic4_note_path = os.path.join(args.mimic4_path, "mimic-iv-note", "3.1", "note")

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

    if args.verbose:
        print(
            f"START:\n\tED STAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
        )

    # Remove emergency department stays that do not result in a hospital admission.
    stays.filter(pl.col("disposition") == "ADMITTED").drop_nulls(subset="hadm_id").drop(
        "disposition"
    )

    if args.verbose:
        print(
            f"REMOVE ED STAYS WITHOUT ADMISSION:\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
        )

    # order matters here ad stays has hadm_id as floats so merge with hadm_id instead
    stays = admits.join(stays, how="inner", on=["subject_id", "hadm_id"])

    # add patient info to admissions data
    stays = stays.join(patients, how="inner", on="subject_id")

    # ensure one ed stay per hospital admission
    stays = m4c.filter_on_nb_stays(stays)

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

    # sample n subjects (can be used to test/speed up processing)
    if args.sample is not None:
        # set the seed for reproducibility
        rng = np.random.default_rng(0)
        stays = stays.sample(n=args.sample, seed=0)

        if args.verbose:
            print(
                f"SELECTING RANDOM SAMPLE OF {args.sample} 'STAYS':\n\tSTAY_IDs: {get_n_unique_values(stays, 'stay_id')}\n\tHADM_IDs: {get_n_unique_values(stays, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(stays)}"
            )

    # Write all stays
    stays.write_csv(os.path.join(args.output_path, "stays.csv"))

    items = (
        set(read_from_txt(args.labitems, as_type="int"))
        if args.labitems
        # see README.md for info on these preselected labevent items
        else [51221, 50912, 51301, 51265, 50971, 50983, 50931, 50893, 50960]
    )

    for table in tqdm(
        args.event_tables, desc="Reading events tables...", total=len(args.event_tables)
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
        )

        if table == "labevents":
            print(
                "Trying to impute missing hadm_ids using join: see https://mimic.mit.edu/docs/iv/modules/hosp/labevents/"
            )
            # use admissions table to impute missing hadm_ids based on charttime
            if not args.lazy:
                admits = admits.lazy()
            events = m4c.get_hadm_id_from_admits(events, admits)

        elif table == "vitalsign":
            # fill in missing hadm_id using stay_id based on stays
            events = impute_from_df(events, stays, "stay_id", "hadm_id")

        if args.verbose:
            print(
                f"{table.upper()}:\n\tHADM_IDs: {get_n_unique_values(events, 'hadm_id')}\n\tTOTAL EVENTS: {events.collect(streaming=True).height}\n\tSUBJECT_IDs: {get_n_unique_values(events)}"
            )

        # Filter by subject_id from stays of interest
        events = events.filter(
            pl.col("subject_id").is_in(
                stays.unique(subset="subject_id").get_column("subject_id").to_list()
            )
        )

        if args.verbose:
            print(
                f"FILTER BY SUBJECT ID:\n\tHADM_IDs: {get_n_unique_values(events, 'hadm_id')}\n\tTOTAL EVENTS: {events.collect(streaming=True).height}\n\tSUBJECT_IDs: {get_n_unique_values(events)}"
            )

        # if hadm_id can't be found then drop since data cannot be assigned
        events = events.drop_nulls(subset="hadm_id")

        if args.verbose:
            print(
                f"REMOVE EVENTS WITH MISSING HADM_ID:\n\tHADM_IDs: {get_n_unique_values(events, 'hadm_id')}\n\tTOTAL EVENTS: {events.collect(streaming=True).height}\n\tSUBJECT_IDs: {get_n_unique_values(events)}"
            )

        # Filter events by hadm_id in stays table (only include relevant events)
        events = events.filter(
            pl.col("hadm_id").is_in(stays.get_column("hadm_id").to_list())
        )

        if args.verbose:
            print(
                f"FILTER BY STAY HADM_IDs:\n\tHADM_IDs: {get_n_unique_values(events, 'hadm_id')}\n\tTOTAL EVENTS: {events.collect(streaming=True).height}\n\tSUBJECT_IDs: {get_n_unique_values(events)}"
            )
        # Collect events
        events = events.collect(streaming=True)

        # Save to disk
        if os.path.exists(os.path.join(args.output_path, "events.csv")):
            with open(os.path.join(args.output_path, "events.csv"), mode="ab") as f:
                events.write_csv(f, include_header=False)
        else:
            events.write_csv(os.path.join(args.output_path, "events.csv"))

    if args.include_notes:
        notes = m4c.read_notes(mimic4_note_path, use_lazy=args.lazy)

        if args.verbose:
            if type(notes) == pl.LazyFrame:
                notes_height = notes.collect(streaming=True).height
            else:
                notes_height = notes.height
            print(
                f"NOTES:\n\tHADM_IDs: {get_n_unique_values(notes, 'hadm_id')}\n\tTOTAL NOTES: {notes_height}\n\tSUBJECT_IDs: {get_n_unique_values(notes)}"
            )

        # Filter notes by hadm_id in stays table
        notes = notes.filter(
            pl.col("hadm_id").is_in(stays.get_column("hadm_id").to_list())
        )

        if args.verbose:
            if type(notes) == pl.LazyFrame:
                notes_height = notes.collect(streaming=True).height
            else:
                notes_height = notes.height
            print(
                f"FILTER BY STAY HADM_IDs:\n\tHADM_IDs: {get_n_unique_values(notes, 'hadm_id')}\n\tTOTAL NOTES: {notes_height}\n\tSUBJECT_IDs: {get_n_unique_values(notes)}"
            )

        # Write all notes

        # if lazy then collect since write_csv is a DataFrame function
        if type(notes) == pl.LazyFrame:
            print("Collecting...")
            notes = notes.collect(streaming=True)

        notes.write_csv(os.path.join(args.output_path, "notes.csv"))

    print("Data extracted.")
