import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import toml
from tqdm import tqdm

from utils.functions import save_pickle
from utils.preprocessing import (
    clean_notes,
    encode_categorical_features,
    extract_lookup_fields,
    generate_interval_dataset,
    generate_train_val_test_set,
    process_text_to_embeddings,
    remove_correlated_features,
)

parser = argparse.ArgumentParser(
    description="Preprocess multimodal MIMIC-IV data for training."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing processed data from extract_data.py.",
)
parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    default="../outputs/processed_data",
    help="Directory to save processed training ids and pkl files.",
)
parser.add_argument(
    "--output_summary_dir",
    "-s",
    type=str,
    default="../outputs/exp_data",
    help="Directory to save summary table file for training/val/test split.",
)
parser.add_argument(
    "--output_reference_dir",
    "-r",
    type=str,
    default="../outputs/reference",
    help="Directory to save lookup table file containing dates and additional fields from EHR data.",
)
parser.add_argument(
    "--pkl_fname",
    type=str,
    default="mmfair_feat.pkl",
    help="Name of pickle file to save processed data.",
)
parser.add_argument(
    "--col_fname",
    type=str,
    default="mmfair_cols.pkl",
    help="Name of pickle file to save column lookup dictionary.",
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    help="Path to config toml file containing lookup fields for grouping.",
    default="../config/targets.toml",
)
parser.add_argument(
    "--corr_threshold",
    type=float,
    default=0.90,
    help="Threshold for removing correlated features. Features with correlation above this value will be removed.",
)
parser.add_argument(
    "--corr_method",
    type=str,
    default="pearson",
    help="Method for removing correlated features. Defaults to Pearson correlation.",
)
parser.add_argument(
    "--min_events", type=int, default=2, help="Minimum number of events per patient."
)
parser.add_argument(
    "--max_events", type=int, default=None, help="Maximum number of events per stay."
)
parser.add_argument(
    "--impute",
    type=str,
    default=None,
    help="Impute strategy. One of ['forward', 'backward', 'mask', 'value' or None]",
)
parser.add_argument(
    "--no_resample",
    action="store_true",
    help="Flag to turn off time-series resampling.",
)
parser.add_argument(
    "--include_dyn_mean",
    action="store_true",
    help="Flag for whether to add mean of dynamic features to static data.",
)
parser.add_argument(
    "--standardize",
    action="store_true",
    help="Flag for whether to standardize timeseries data with minmax scaling (in the range [0,1]).",
)
parser.add_argument(
    "--max_elapsed",
    type=int,
    default=72,
    help="Max time elapsed from hospital admission (hours). Filters any events that occur after this.",
)
parser.add_argument(
    "--include_notes",
    action="store_true",
    help="Whether to preprocess notes if available.",
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="Ratio of training data to create split.",
)
parser.add_argument(
    "--stratify",
    action="store_true",
    default=False,
    help="Whether to stratify the split by outcome and sensitive attributes.",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for random sampling. Defaults to 0."
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")
args = parser.parse_args()
output_dir = args.data_dir if args.output_dir is None else args.output_dir

print(f"Processing data from {args.data_dir}...")

# If pkl file exists then remove and start over
if len(glob.glob(os.path.join(output_dir, "*.pkl"))) > 0:
    response = input("Will need to overwrite existing data... continue? (y/n)")
    if response == "y":
        for f in glob.glob(os.path.join(output_dir, "*.pkl")):
            try:
                os.remove(f)
            except OSError as ex:
                print(ex)
                sys.exit()
    else:
        print("Exiting..")
        sys.exit()

elif not os.path.exists(output_dir):
    print(f"Creating directory at {output_dir}...")
    os.makedirs(output_dir)

print(f"Reading pre-extracted data from {args.data_dir}...")
# Read extracted data
if os.path.exists(os.path.join(args.data_dir, "ehr_static.csv")):
    ehr_data = pl.read_csv(
        os.path.join(args.data_dir, "ehr_static.csv"), try_parse_dates=True
    )
else:
    print(f"No EHR data found under {args.data_dir}. Exiting..")
    sys.exit()
if os.path.exists(os.path.join(args.data_dir, "events_ts.csv")):
    events = pl.scan_csv(
        os.path.join(args.data_dir, "events_ts.csv"), try_parse_dates=True
    )
else:
    print(f"No time-series data found under {args.data_dir}. Exiting..")
    sys.exit()
if args.include_notes and os.path.exists(os.path.join(args.data_dir, "notes.csv")):
    notes = pl.scan_csv(os.path.join(args.data_dir, "notes.csv"))

#### TRAIN-VAL-TEST SPLIT ####
print("Splitting data into training, validation and test sets..")
config = toml.load(args.config)
outcomes = config["outcomes"]["labels"]
lookup = config["attributes"]["lookup"]
vitals_freq = config["timeseries"]["vitals_freq"]
lab_freq = config["timeseries"]["lab_freq"]
vitals = config["attributes"]["vitals"]
## Features to save within EHR data (not candidates for exclusion due to high correlation)
feats_to_save = config["attributes"]["feats_to_save"]

print("---------------------------------")
print("START STATIC DATA PREPROCESSING")
print("---------------------------------")
#### STATIC DATA PREPROCESSING ####
if args.verbose:
    print("Preprocessing static EHR data for training..")
    ehr_proc = encode_categorical_features(ehr_data)
    ### Save edregtime for time-series processing
    ehr_regtime = ehr_proc.select(["subject_id", "edregtime"])
    ehr_proc = extract_lookup_fields(
        ehr_proc, lookup, lookup_output_path=args.output_reference_dir
    )
    ehr_proc = remove_correlated_features(
        ehr_proc,
        feats_to_save,
        threshold=args.corr_threshold,
        method=args.corr_method,
        verbose=args.verbose,
    )
print("---------------------------------")
#### TIMESERIES PREPROCESSING ####
print("START TIME-SERIES DATA PREPROCESSING")
print("---------------------------------")
# collect events
events = events.collect(streaming=True)
# get all features expected for each event data source and set sampling freq
print(f"Imputing missing values using strategy: {args.impute}")
feature_dict, col_dict = generate_interval_dataset(
    ehr_proc,
    events,
    ehr_regtime,
    vitals_freq,
    lab_freq,
    args.min_events,
    args.max_events,
    args.impute,
    args.include_dyn_mean,
    args.no_resample,
    args.standardize,
    args.max_elapsed,
    vitals,
    outcomes,
    args.verbose,
)
print("---------------------------------")
#### NOTES PREPROCESSING ###
if args.include_notes:
    print("START NOTES DATA PREPROCESSING")
    if args.verbose:
        print("Parsing discharge notes features from BHC segment..")
    notes = notes.select(["subject_id", "target"]).cast(
        {"subject_id": pl.Int64, "target": pl.String}
    )
    ### Select only notes with captured measurements
    notes = notes.filter(pl.col("subject_id").is_in(feature_dict.keys()))
    # Clean notes by removing "___" identifiers
    if args.verbose:
        print("Cleaning discharge notes from extra identifiers..")
    notes = clean_notes(notes).collect(streaming=True)
    # Generate embeddings
    if args.verbose:
        print("Generating sentence-level embeddings for discharge notes..")
    embeddings = process_text_to_embeddings(notes)
    # Add embeddings to feature dictionary
    if args.verbose:
        print("Appending embeddings to feature dictionary..")
    for id_val in tqdm(
        notes.unique("subject_id").get_column("subject_id").to_list(),
        desc="Linking note embeddings to feature dictionary...",
    ):
        if id_val not in embeddings.keys() or id_val not in feature_dict.keys():
            continue
        feature_dict[id_val]["notes"] = embeddings[id_val]

print("START TRAIN-VAL-TEST SPLIT")
print("---------------------------------")
#### TRAIN-VAL-TEST SPLIT ####
for outcome in outcomes:
    print(f"Processing splits for outcome: {outcome}")
    ehr_data = ehr_data.filter(pl.col("subject_id").is_in(feature_dict.keys()))
    train_dict = generate_train_val_test_set(
        ehr_data,
        args.output_dir,
        outcome,
        args.output_summary_dir,
        args.seed,
        args.train_ratio,
        (1 - args.train_ratio) / 2,
        (1 - args.train_ratio) / 2,
        stratify=args.stratify,
        verbose=args.verbose,
    )

### Add scaled feature variants to feature dictionary
print("Adding scaled feature variants to feature dictionary..")
prep_data = pd.concat([train_dict["train"], train_dict["val"], train_dict["test"]])
prep_data = encode_categorical_features(pl.DataFrame(prep_data))
prep_data = prep_data.to_pandas()
static_cols = col_dict["static_cols"]
dynamic0_cols = col_dict["dynamic0_cols"]
dynamic1_cols = col_dict["dynamic1_cols"]

# Create a mapping of subject_id to its corresponding scaled features
scaled_features_map = prep_data[["subject_id"] + static_cols]
scaled_features_map = scaled_features_map.set_index("subject_id").to_dict(
    orient="index"
)

for p_id in tqdm(feature_dict.keys()):
    # print(scaled_features_map[p_id])
    feature_dict[p_id]["static"] = (
        np.array(list(scaled_features_map[p_id].values())).round(5).astype(np.float32)
    )
    feature_dict[p_id]["static"] = feature_dict[p_id]["static"].reshape(1, -1)

print("---------------------------------")
print("Finished train/val/test split creation.")
print("---------------------------------")
# Preview example data
# example_id = list(data_dict.keys())[-1]
# print(f"Example data:\n\t{data_dict[example_id]}")
print(f"Feature preparation successful. Exporting prepared features to {output_dir}..")
# Save dictionary to disk
save_pickle(feature_dict, output_dir, args.pkl_fname)
save_pickle(col_dict, output_dir, args.col_fname)
print("Finished feature preparation for multimodal learning.")
