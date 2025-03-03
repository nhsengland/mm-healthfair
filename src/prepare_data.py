import argparse
import glob
import os
import pickle
import sys

import polars as pl
from tqdm import tqdm

from utils.functions import scale_numeric_features
from utils.preprocessing import (
    add_time_elapsed_to_events,
    clean_events,
    clean_notes,
    convert_events_to_timeseries,
    encode_categorical_features,
    process_text_to_embeddings,
)

parser = argparse.ArgumentParser(
    description="Preprocess data - generates pkl file to use for training."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing processed data, will be used to output processed pkl file.",
)
parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    help="Directory to save processed dictionary file and outputs.",
)
parser.add_argument(
    "--min_events", type=int, default=2, help="Minimum number of events per stay."
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
    "--no_scale", action="store_true", help="Flag to turn off feature scaling."
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
    "--max_elapsed",
    type=int,
    default=48,
    help="Max time elapsed from hospital admission (hours). Filters any events that occur after this.",
)
parser.add_argument(
    "--include_notes",
    action="store_true",
    help="Whether to preprocess notes if available.",
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

args = parser.parse_args()
output_dir = args.data_dir if args.output_dir is None else args.output_dir

print(
    f"Processing data from {args.data_dir} and saving output files to {output_dir}..."
)

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

# Read extracted data
stays = pl.scan_csv(os.path.join(args.data_dir, "stays.csv"))
events = pl.scan_csv(os.path.join(args.data_dir, "events.csv"), try_parse_dates=True)

# Check if notes exists if so read csv
if os.path.exists(os.path.join(args.data_dir, "notes.csv")) and args.include_notes:
    with_notes = True
    notes = pl.scan_csv(os.path.join(args.data_dir, "notes.csv"))
else:
    with_notes = False

#### STATIC DATA PREPROCESSING ####

metadata = stays.collect()

static_features = [
    "anchor_age",
    "gender",
    "race",
    "marital_status",
    "insurance",
    "los",
    "los_ed",
]

# Select features of interest only
static_data = metadata.select(["hadm_id"] + static_features).cast(
    {"los": pl.Float64, "los_ed": pl.Float64}
)

# Applies min max scaling to  numerical features
numeric_cols = ["anchor_age", "height", "weight", "los_ed"]

if not args.no_scale:
    static_data = scale_numeric_features(
        static_data, numeric_cols=[i for i in static_data.columns if i in numeric_cols]
    )
static_data = encode_categorical_features(static_data)

#### NOTES PREPROCESSING ###

if with_notes:
    notes = notes.select(["hadm_id", "charttime", "text"]).cast(
        {"hadm_id": pl.Int64, "charttime": pl.Datetime, "text": pl.String}
    )
    # Extract specific section of notes "Brief Hospital Course"
    # Drops ~3k missing the section entirely
    notes = notes.with_columns(
        (pl.col("text").str.find("History of Present Illness:")).alias("begin")
    ).drop_nulls(subset="begin")
    # Get the end
    # Drops any remaining where the end cant be found (~450)
    notes = notes.with_columns(
        (
            pl.col("text")
            .str.slice(pl.col("begin"))
            .str.find("\n \nPast Medical History")
        ).alias("end")
    ).drop_nulls(subset="end")

    # Get subsection
    # NOTE: This is not perfect and in some cases snippet may contain text from other preceding sections

    notes = notes.with_columns(
        subtext=pl.col("text").str.slice(pl.col("begin") + 27, pl.col("end"))
    ).drop(["text", "begin", "end"])

    # Clean notes by removing "___" identifiers
    notes = clean_notes(notes).collect(streaming=True)

    # Generate embeddings
    embeddings = process_text_to_embeddings(notes)

#### TIMESERIES PREPROCESSING ####

# clean events
events = clean_events(events)

# collect events
events = events.collect(streaming=True)

# scale values from events data
if not args.no_scale:
    events = scale_numeric_features(events, ["value"], over="label")

### CREATE DICTIONARY DATA

data_dict = {}

# Filter events by number of events per stay
min_events = 1 if args.min_events is None else int(args.min_events)
max_events = 1e6 if args.max_events is None else int(args.max_events)

# get number of different event sources
n_src = events.n_unique("linksto")

# Loop over events per stay to generate a key-data structure
print(f"Imputing missing values using strategy: {args.impute}")
n = 0
filter_by_nb_events = 0
missing_event_src = 0
filter_by_elapsed_time = 0
missing_notes = 0

# get all features expected for each event data source and set sampling freq
feature_map = {}
freq = {}
for src in events.unique("linksto").get_column("linksto").to_list():
    feature_map[src] = sorted(
        events.filter(pl.col("linksto") == src)
        .unique("label")
        .get_column("label")
        .to_list()
    )
    freq[src] = "30m" if src == "vitalsign" else "5h"


for stay_events in tqdm(
    events.partition_by("hadm_id", include_key=True),
    desc="Generating stay-level data...",
):
    id_val = stay_events["hadm_id"][0]

    # Get static data for stay
    stay_static = static_data.filter(pl.col("hadm_id") == id_val).drop("hadm_id")

    if stay_static.shape[0] == 0:
        # skip if not in stays
        continue

    # Get metadata (unprocessed static data)
    stay_metadata = metadata.filter(pl.col("hadm_id") == id_val)

    admittime = stay_metadata.select("admittime").cast(pl.Datetime).item()

    if with_notes:
        dischtime = stay_metadata.select("dischtime").cast(pl.Datetime).item()

        # Get discharge notes relating to hospital stay
        stay_notes = notes.filter(pl.col("hadm_id") == id_val).drop("hadm_id")

        # Ensure that notes are charted during hospital stay
        # TODO: Change to first x hours after admission??
        stay_notes = stay_notes.filter(
            (pl.col("charttime") >= admittime) & (pl.col("charttime") <= dischtime)
        )

        if stay_notes.shape[0] == 0:
            missing_notes += 1
            # skip if no notes for stay or within hospital stay
            continue

    # filter if not at least one entry from each event data source
    if stay_events.n_unique("linksto") < n_src:
        missing_event_src += 1
        continue

    write_data = True
    ts_data = []
    for events_by_src in stay_events.partition_by("linksto"):
        src = events_by_src.select(pl.first("linksto")).item()

        # Convert event data to timeseries
        timeseries = convert_events_to_timeseries(events_by_src)

        if (timeseries.shape[0] < min_events) | (timeseries.shape[0] > max_events):
            filter_by_nb_events += 1
            write_data = False
            break

        features = feature_map[src]
        # Ensure models have the same number of features
        missing_cols = [x for x in features if x not in timeseries.columns]
        # create empty columns for missing features
        timeseries = timeseries.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing_cols]
        )

        # Impute missing values
        if args.impute is not None:
            # TODO: Consider using mean value for missing static data such as height and weight rather than constant?

            if args.impute == "mask":
                # Add new mask columns for whether row is nan or not
                timeseries = timeseries.with_columns(
                    [pl.col(f).is_null().alias(f + "_isna") for f in features]
                )
                stay_static = stay_static.with_columns(
                    [
                        pl.col(f).is_null().alias(f + "_isna")
                        for f in stay_static.columns
                    ]
                )

            elif (args.impute == "forward") | (args.impute == "backward"):
                # Fill missing values using forward fill
                timeseries = timeseries.fill_null(strategy=args.impute)

                # for remaining null values use fixed value
                timeseries = timeseries.fill_null(value=-1)
                stay_static = stay_static.fill_null(value=-1)

            elif args.impute == "value":
                timeseries = timeseries.fill_null(value=-1)
                stay_static = stay_static.fill_null(value=-1)

            else:
                raise ValueError(
                    "impute_strategy must be one of [None, mask, value, forward, backward]"
                )

        if args.include_dyn_mean:
            # Option to get mean value during stay (drop time col)
            timeseries_mean = timeseries.drop(["charttime", "linksto"]).mean()
            timeseries_mean = timeseries_mean.with_columns(pl.all().round(3))
            # Add to static data
            stay_static = stay_static.hstack(timeseries_mean)

        if not args.no_resample:
            # Upsample and then downsample to create regular intervals e.g., 2-hours
            timeseries = timeseries.upsample(time_column="charttime", every="1m")
            timeseries = timeseries.group_by_dynamic(
                "charttime",
                every=freq[src],
            ).agg(pl.col(pl.Float64).mean())
            timeseries = timeseries.fill_null(strategy="forward")

        timeseries = add_time_elapsed_to_events(timeseries, admittime)
        # only include first x hours - note this could lead to all data being lost so skip if that is the case
        timeseries = timeseries.filter(pl.col("elapsed") <= args.max_elapsed)

        if timeseries.shape[0] == 0:
            filter_by_elapsed_time += 1
            write_data = False
            break

        timeseries = timeseries.select(features)

        ts_data.append(timeseries)

    if write_data:
        data_dict[id_val] = {}
        data_dict[id_val]["static"] = stay_static

        for idx, ts in enumerate(ts_data):
            data_dict[id_val][f"dynamic_{idx}"] = ts

        if with_notes:
            data_dict[id_val]["notes"] = embeddings[id_val]
        n += 1

    write_data = True

print(f"SUCCESSFULLY PROCESSED DATA FOR {n} STAYS.")
print(f"SKIPPING {filter_by_nb_events} STAYS DUE TO TOTAL NUM EVENTS.")
print(f"SKIPPING {missing_event_src} STAYS DUE TO MISSING EVENT SOURCE.")
print(f"SKIPPING {filter_by_elapsed_time} STAYS DUE TO FILTER ON ELAPSED TIME.")
if with_notes:
    print(f"SKIPPING {missing_notes} STAYS DUE TO MISSING NOTES.")

# Preview example data
example_id = list(data_dict.keys())[-1]
print(f"Example data:\n\t{data_dict[example_id]}")

# Save dictionary to disk
with open(os.path.join(output_dir, "processed_data.pkl"), "wb") as f:
    pickle.dump(data_dict, f)
print("Finished.")
