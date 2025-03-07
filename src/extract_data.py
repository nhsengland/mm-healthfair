import argparse  # noqa: I001
import os
import sys
import shutil
import gzip
import numpy as np
import polars as pl
from tqdm import tqdm

import utils.mimiciv as m4c
from utils.preprocessing import preproc_icd_module, get_ltc_features
from utils.functions import get_n_unique_values, impute_from_df, get_final_episodes, get_demographics_summary

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
    parser.add_argument("--include_events", "-t", action="store_true")
    parser.add_argument("--include_addon_ehr_data", '-a', action="store_true")

    parser.add_argument(
        "--labitems",
        "-i",
        type=str,
        help="Text file containing list of ITEMIDs to use from labevents.",
    )
    parser.add_argument(
        "--icd9_to_icd10",
        type=str,
        help="Text file containing ICD 9-10 mapping.",
    )
    parser.add_argument(
        "--ltc_mapping",
        type=str,
        help="JSON file containing mapping for long-term conditions in ICD-10 format.",
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
        "--top_n_medications",
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
    admits = m4c.read_admissions_table(mimic4_path, use_lazy=args.lazy)
    if args.verbose:
        print(
            f"START:\n\tInitial stays with ED attendance: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
    patients = m4c.read_patients_table(mimic4_path, use_lazy=args.lazy)
    if args.verbose:
        print(
            f"\n\tValidated stays at age>=18: {get_n_unique_values(patients, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(patients)}"
        )
        print("Creating patient-level dataset based on final episodes...")
    # Get final ED episode with hospitalisation to reduce processing time
    admits_last = get_final_episodes(admits)
    admits_last = m4c.read_icu_table(mimic4_icu_path, admits, use_lazy=args.lazy)

    # Process long-term conditions
    diagnoses = m4c.read_diagnoses_table(mimic4_path, admits, use_lazy=args.lazy)
    if args.verbose:
        print(
            f"DIAGNOSES:\n\tUnique ICD-10 conditions across stays: {get_n_unique_values(diagnoses, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(diagnoses)}"
        )
    diagnoses = preproc_icd_module(diagnoses, icd_map_path=args.icd9_to_icd10, 
                                   ltc_dict_path=args.ltc_mapping, verbose=args.verbose,
                                   use_lazy=args.lazy)
    admits_last = get_ltc_features(admits_last, diagnoses, ltc_dict_path=args.ltc_mapping,
                                   use_lazy=args.lazy)
    if args.verbose:
        print("Printing characteristics in full patient sample.")
        get_demographics_summary(admits_last)
    
    ### Optional random sampling to understample subjects
    # sample n subjects (can be used to test/speed up processing)
    if args.sample is not None:
        if args.verbose:
            print(
                f"SELECTING RANDOM SAMPLE OF {args.sample} 'PATIENTS WITH ED ATTENDANCE'."
            )
        # set the seed for reproducibility
        rng = np.random.default_rng(0)
        admits_last = admits_last.sample(n=args.sample, seed=0)
        if args.verbose:
            print("Printing characteristics in random sample.")
            get_demographics_summary(admits_last)

    if args.include_notes:
        notes = m4c.read_notes(admits, admits_last, mimic4_note_path, 
                               verbose=args.verbose, use_lazy=args.lazy)
        admits_last, notes = m4c.get_notes_population(notes, admits_last, verbose=args.verbose,
                                            use_lazy=args.lazy)
        if args.verbose:
            print(
                f"NOTES:\n\tUnique patients with notes history: {get_n_unique_values(admits_last)} with median {notes['num_summaries'].median()} notes per patient (IQR: {notes['num_summaries'].quantile(0.25)} - {notes['num_summaries'].quantile(0.75)})."
            )
    
    if args.include_events:
        print('Getting time-series data from OMR table..')
        omr = m4c.read_omr_table(mimic4_path, admits_last, use_lazy=args.lazy)
        print('Getting time-series data from ED vital signs table..')
        ed_vitals = m4c.read_vitals_table(mimic4_ed_path, admits_last, use_lazy=args.lazy)
        print('Getting lab test measures..')
        
        # read compressed and write to file since lazy polars API can only scan uncompressed csv's
        if not os.path.exists(os.path.join(mimic4_path, f"labevents.csv")):
            print(f"Uncompressing labevents data... (required)")
            with gzip.open(os.path.join(mimic4_path, f"labevents.csv.gz"), "rb") as f_in:
                with open(os.path.join(mimic4_path, f"labevents.csv"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        labs = m4c.read_labevents_table(mimic4_path, admits_last, include_items=args.labitems)
        print('Merging OMR, ED and Lab test measurements..')
        events = m4c.merge_events_table(ed_vitals, labs, omr, use_lazy=args.lazy)
        print('Filtering population with ED attendance, discharge summary and measurements history..')
        admits_last = m4c.get_population_with_measures(events, admits_last, use_lazy=args.lazy)
        print(
                f"TIME-SERIES:\n\tUnique patients with recorded measurements: {get_n_unique_values(admits_last)} with median {admits_last['num_measures'].median()} measurements per patient (IQR: {admits_last['num_summaries'].quantile(0.25)} - {admits_last['num_summaries'].quantile(0.75)})."
            )

    if args.include_addon_ehr_data:
        print('Parsing additional medication and specialty data from the EHR..')
        admits_last = m4c.read_medications_table(mimic4_path, admits_last, use_lazy=args.lazy,
                                                 top_n=args.top_n_medications)
        print(
                f"MEDICATIONS (EHR):\n\tParsed medication history with median {admits_last['total_n_presc'].median()} administered drugs per patient (IQR: {admits_last['total_n_presc'].quantile(0.25)} - {admits_last['total_n_presc'].quantile(0.75)})."
            )
        print('Getting specialty data..')
        admits_last = m4c.read_specialty_table(mimic4_path, admits_last, use_lazy=args.lazy)
        print(
                f"SPECIALTIES (EHR):\n\tParsed order history with median {admits_last['total_proc_count'].median()} administered drugs per patient (IQR: {admits_last['total_proc_count'].quantile(0.25)} - {admits_last['total_proc_count'].quantile(0.75)})."
            )
        
    if args.verbose:
        print("Completed data extraction.")
        print("Writing data to disk..")
        if args.include_events and args.include_notes:
            m4c.save_multimodal_dataset(admits_last, notes, events, output_path=args.output_path)
        elif args.include_events:
            m4c.save_multimodal_dataset(admits_last, admits_last, events, use_notes=False, output_path=args.output_path)
        elif args.include_notes:
            m4c.save_multimodal_dataset(admits_last, notes, admits_last, use_events=False, output_path=args.output_path)
        else:
            m4c.save_multimodal_dataset(admits_last, admits_last, admits_last, use_events=False, use_notes=False, output_path=args.output_path)
        print(f"Exported extracted MIMIC-IV data to CSV files to {args.output_path}.")

    print("Data extraction complete.")
