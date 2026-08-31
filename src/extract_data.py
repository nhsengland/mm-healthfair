import argparse
import gzip
import os
import shutil
import sys
import warnings

import numpy as np
import polars as pl

import utils.mimiciv as m4c
from utils.functions import (
    get_demographics_summary,
    get_final_episodes,
    get_n_unique_values,
)
from utils.preprocessing import get_ltc_features, preproc_icd_module

warnings.filterwarnings("ignore", category=pl.exceptions.MapWithoutReturnDtypeWarning)

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

    parser.add_argument("--include_notes", "-n", default=False, action="store_true")
    parser.add_argument("--include_events", "-t", default=False, action="store_true")
    parser.add_argument(
        "--include_addon_ehr_data", "-a", default=False, action="store_true"
    )

    parser.add_argument(
        "--labitems",
        "-i",
        type=str,
        help="Text file containing list of ITEMIDs to use from labevents.",
        default="../config/lab_items.txt",
    )
    parser.add_argument(
        "--icd9_to_icd10",
        type=str,
        help="Text file containing ICD 9-10 mapping.",
        default="../config/icd9to10.txt",
    )
    parser.add_argument(
        "--ltc_mapping",
        type=str,
        help="JSON file containing mapping for long-term conditions in ICD-10 format.",
        default="../config/ltc_mapping.json",
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
        help="Extract smaller patient sample (random).",
    )
    parser.add_argument(
        "--top_n_meds",
        type=int,
        help="Number of drug-level features to extract (if using add-on EHR data).",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False (except for events tables - always uses lazymode).",
    )

    args, _ = parser.parse_known_args()
    mimic4_path = os.path.join(args.mimic4_path, "mimiciv", "3.1", "hosp")
    mimic4_ed_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "3.1", "ed")
    mimic4_icu_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "3.1", "icu")
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
    admits = m4c.read_admissions_table(
        mimic4_path, use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(
            f"START:\n\tInitial stays with ED attendance: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
    admits = m4c.read_patients_table(mimic4_path, admits, use_lazy=args.lazy)
    if args.verbose:
        print(
            f"\n\tValidated stays at age>=18: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
        print("Creating patient-level dataset based on final episodes...")
    # Get final ED episode with hospitalisation to reduce processing time
    admits_icu = m4c.read_icu_table(
        mimic4_icu_path, admits, use_lazy=args.lazy, verbose=args.verbose
    )
    admits_last = get_final_episodes(admits_icu)

    ### Optional random sampling to understample subjects
    # sample n subjects (can be used to test/speed up processing)
    if args.sample is not None:
        if args.verbose:
            print(
                f"SELECTING RANDOM SAMPLE OF {args.sample} PATIENTS WITH ED ATTENDANCE."
            )
        # set the seed for reproducibility
        rng = np.random.default_rng(0)
        if isinstance(admits_last, pl.LazyFrame):
            admits_last = admits_last.collect()
        admits_last = admits_last.sample(n=args.sample, seed=0)

    # Process long-term conditions
    diagnoses = m4c.read_diagnoses_table(
        mimic4_path, admits_icu, admits_last, use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(
            f"DIAGNOSES:\n\tUnique ICD-10 conditions across stays: {get_n_unique_values(diagnoses, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(diagnoses)}"
        )
    diagnoses = preproc_icd_module(
        diagnoses,
        icd_map_path=args.icd9_to_icd10,
        ltc_dict_path=args.ltc_mapping,
        verbose=args.verbose,
        use_lazy=args.lazy,
    )
    admits_last = get_ltc_features(
        admits_last, diagnoses, ltc_dict_path=args.ltc_mapping, use_lazy=args.lazy
    )

    if args.verbose:
        print("------------------------------------------")
        print("Printing characteristics in full patient sample.")
        get_demographics_summary(admits_last)
        print("------------------------------------------")

    if args.include_notes:
        notes = m4c.read_notes(
            admits_icu,
            admits_last,
            mimic4_note_path,
            verbose=args.verbose,
            use_lazy=args.lazy,
        )
        admits_last, notes = m4c.get_notes_population(
            notes, admits_last, use_lazy=args.lazy
        )
        if args.verbose:
            summaries = notes.collect().to_pandas()["num_summaries"]
            print(
                f"NOTES:\n\tUnique patients with notes history: {get_n_unique_values(admits_last)} with median {summaries.median()} notes per patient (IQR: {summaries.quantile(0.25)} - {summaries.quantile(0.75)})."
            )

    if args.include_events:
        print("Getting time-series data from OMR table..")
        omr = m4c.read_omr_table(mimic4_path, admits_last, use_lazy=args.lazy)
        print("Getting time-series data from ED vital signs table..")
        ed_vitals = m4c.read_vitals_table(
            mimic4_ed_path, admits_last, use_lazy=args.lazy
        )
        print("Getting lab test measures..")

        # read compressed and write to file since lazy polars API can only scan uncompressed csv's
        if not os.path.exists(os.path.join(mimic4_path, "labevents.csv")):
            print("Uncompressing labevents data... (required)")
            with gzip.open(os.path.join(mimic4_path, "labevents.csv.gz"), "rb") as f_in:
                with open(os.path.join(mimic4_path, "labevents.csv"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        labs = m4c.read_labevents_table(
            mimic4_path, admits_last, include_items=args.labitems
        )
        print("Merging OMR, ED and Lab test measurements..")
        events = m4c.merge_events_table(ed_vitals, labs, omr, use_lazy=args.lazy)
        print(
            "Filtering population with ED attendance, discharge summary and measurements history.."
        )
        admits_last = m4c.get_population_with_measures(
            events, admits_last, use_lazy=args.lazy
        )
        measures = admits_last.collect().to_pandas()["num_measures"]
        print(
            f"TIME-SERIES:\n\tUnique patients with recorded measurements: {get_n_unique_values(admits_last)} with median {measures.median()} measurements per patient (IQR: {measures.quantile(0.25)} - {measures.quantile(0.75)})."
        )
        ### Filter subject_ids that are in admits_last
        if args.include_notes:
            notes = notes.collect().filter(
                pl.col("subject_id").is_in(admits_last.collect().select("subject_id"))
            )

    if args.include_addon_ehr_data:
        print("Parsing additional medication and specialty data from the EHR..")
        admits_last = m4c.read_medications_table(
            mimic4_path, admits_last, use_lazy=args.lazy, top_n=args.top_n_meds
        )
        meds = admits_last.collect().to_pandas()["total_n_presc"]
        print(
            f"MEDICATIONS (EHR):\n\tParsed medication history with median {meds.median()} administered drugs per patient (IQR: {meds.quantile(0.25)} - {meds.quantile(0.75)})."
        )
        print("Getting specialty data..")
        admits_last = m4c.read_specialty_table(
            mimic4_path, admits_last, use_lazy=args.lazy
        )
        specs = admits_last.collect().to_pandas()["total_proc_count"]
        print(
            f"SPECIALTIES (EHR):\n\tParsed order history with median {specs.median()} provider orders per patient (IQR: {specs.quantile(0.25)} - {specs.quantile(0.75)})."
        )

    if args.verbose:
        print("Completed data extraction.")
        print("Writing data to disk..")
        if args.include_events and args.include_notes:
            m4c.save_multimodal_dataset(
                admits_last, events, notes, output_path=args.output_path
            )
        elif args.include_events:
            m4c.save_multimodal_dataset(
                admits_last,
                events,
                admits_last,
                use_notes=False,
                output_path=args.output_path,
            )
        elif args.include_notes:
            m4c.save_multimodal_dataset(
                admits_last,
                admits_last,
                notes,
                use_events=False,
                output_path=args.output_path,
            )
        else:
            m4c.save_multimodal_dataset(
                admits_last,
                admits_last,
                admits_last,
                use_events=False,
                use_notes=False,
                output_path=args.output_path,
            )
        print(f"Exported extracted MIMIC-IV data as CSV files to {args.output_path}.")

    print("Data extraction complete.")
