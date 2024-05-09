import argparse
import glob
import os
import pickle
import sys

import polars as pl
from tqdm import tqdm
from utils.preprocessing import (
    add_time_elapsed_to_events,
    clean_events,
    convert_events_to_timeseries,
    impute_from_df,
    process_demographic_data,
)

parser = argparse.ArgumentParser(
    description="Preprocess data - generates parquet files to use for training."
)
parser.add_argument(
    "data_path",
    type=str,
    help="Directory containing processed data, will be used to output processed parquet file.",
)
parser.add_argument(
    "--min_events", type=int, default=5, help="Minimum number of events per stay."
)
parser.add_argument(
    "--max_events", type=int, default=None, help="Maximum number of events per stay."
)
parser.add_argument(
    "--impute",
    type=str,
    default=None,
    help="Impute strategy. One of ['forward', 'backward', 'mask', 'value']",
)

parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

args = parser.parse_args()

print("PROCESSING DATA...")

# If episodes exist then remove and start over
if len(glob.glob(os.path.join(args.data_path, "*", "*.parquet"))) > 0:
    response = input("Will need to overwrite existing data... continue? (y/n)")
    if response == "y":
        for f in glob.glob(os.path.join(args.data_path, "*", "*.parquet")):
            try:
                os.remove(f)
            except OSError as ex:
                print(ex)
                sys.exit()
    else:
        print("Exiting..")
        sys.exit()

# read extracted data
stays = pl.scan_csv(os.path.join(args.data_path, "stays.csv"))
events = pl.scan_csv(os.path.join(args.data_path, "events.csv"), try_parse_dates=True)

#### STATIC DATA PREPROCESSING ####

static_features = [
    "anchor_age",
    "gender",
    "race",
    "marital_status",
    "insurance",
    "los",
    "los_ed",
    "mortality",
    "height",
    "weight",
]
static_data = process_demographic_data(stays, features=static_features)

# Filter stays by number of stays per subject
# stays = stays.filter(pl.col("stay_id").count().over('subject_id') >= args.min_stays)

#### TIMESERIES PREPROCESSING ####

# fill in missing hadm_id from stay table
events = impute_from_df(events, stays, "stay_id", "hadm_id").cast(
    {"hadm_id": pl.Int64()}
)

n_no_identifier = (
    events.collect()
    .filter((pl.col.hadm_id.is_null()) & pl.col.stay_id.is_null())
    .shape[0]
)
# if both hadm_id and stay_id still can't be found then drop since data cannot be assigned
print(f"REMOVING {n_no_identifier} EVENTS WITH NO IDENTIFIER (STAY_ID OR HADM_ID)")
events = events.drop_nulls(subset="hadm_id").drop("stay_id")

# clean events
events = clean_events(events)

# collect events
events = events.collect(streaming=True)

# get all features that appear in events data
features = sorted(events.unique(subset="label").get_column("label").to_list())

### CREATE DICTIONARY DATA

data_dict = {}

# Filter events by number of events per stay
min_events = 1 if args.min_events is None else int(args.min_events)
max_events = 1e6 if args.max_events is None else int(args.max_events)

# Loop over events per stay to generate a key-data structure
print(f"Imputing missing values using strategy: {args.impute}")
n = 0
filter_by_nb_events = 0

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

    # Convert event data to timeseries
    timeseries = convert_events_to_timeseries(stay_events)
    timeseries = add_time_elapsed_to_events(timeseries)

    if (timeseries.shape[0] < min_events) | (timeseries.shape[0] > max_events):
        filter_by_nb_events += 1
        continue

    # Ensure models have the same number of features
    missing_cols = [x for x in features if x not in timeseries.columns]
    # create empty columns for missing features
    timeseries = timeseries.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing_cols]
    )
    timeseries = timeseries.select(["charttime", "elapsed"] + features)

    # Impute missing values
    if args.impute is not None:
        # TODO: Consider using mean value for missing static data

        if args.impute == "mask":
            # Add new mask columns for whether row is nan or not
            timeseries = timeseries.with_columns(
                [pl.col(i).is_null().alias(i + "_isna") for i in features]
            )
            stay_static = stay_static.with_columns(
                [pl.col(i).is_null().alias(i + "_isna") for i in static_data.columns]
            )

        elif (args.impute == "forward") | (args.impute == "backward"):
            # Fill missing values using forward fill
            timeseries = timeseries.fill_null(strategy=args.impute)

            # for remaining null values use -999
            timeseries = timeseries.fill_null(value=-999)
            stay_static = stay_static.fill_null(value=-999)

        elif args.impute == "value":
            timeseries = timeseries.fill_null(value=-999)
            stay_static = stay_static.fill_null(value=-999)

        else:
            raise ValueError(
                "impute_strategy must be one of [None, mask, value, forward, backward]"
            )

    # convert data to dict and write to file
    # TODO: Add notes
    data_dict[id_val] = {"static": stay_static, "dynamic": timeseries}
    n += 1

print(f"SUCCESSFULLY PROCESSED DATA FOR {n} STAYS.")
print(f"SKIPPING {filter_by_nb_events} STAYS DUE TO NUM EVENTS.")

example_id = list(data_dict.keys())[-1]
print(
    f"Example data:\n\tStatic: {data_dict[example_id]['static']}\n\tDynamic: {data_dict[example_id]['dynamic']}"
)
# Save dictionary to disk
with open(os.path.join(args.data_path, "processed_data.pkl"), "wb") as f:
    pickle.dump(data_dict, f)
