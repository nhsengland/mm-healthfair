import argparse
import glob
import os
import sys

import polars as pl
from tqdm import tqdm
from utils.preprocessing import (
    assemble_episodic_data,
    clean_events,
)
from utils.subject import (
    add_hours_elapsed_to_events,
    convert_events_to_timeseries,
    get_events_in_period,
    read_events,
    read_stays,
)

parser = argparse.ArgumentParser(description="Extract episodes from per-subject data.")
parser.add_argument(
    "subjects_root_path", type=str, help="Directory containing subject sub-directories."
)
parser.add_argument(
    "mimic4_path", type=str, help="Directory containing MIMIC-IV HOSP CSV files."
)
parser.add_argument(
    "--output_dir", "-o", type=str, help="Optional new directory to save episodic data."
)

parser.add_argument(
    "--min_stays", type=int, default=1, help="Minimum number of stays per subject."
)
parser.add_argument(
    "--min_events", type=int, default=5, help="Minimum number of events per stay."
)
parser.add_argument(
    "--max_events", type=int, default=None, help="Maximum number of events per stay."
)
parser.add_argument(
    "--subject_list",
    type=str,
    help="File containing list of subjects to include.",
)
parser.add_argument(
    "--features",
    type=str,
    help="File containing list of feature names to expect.",
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

args = parser.parse_args()

if args.subject_list is not None:
    subject_list = (
        pl.read_csv(args.subject_list, has_header=False).cast(pl.String).to_list()
    )
else:
    subject_list = [
        name
        for name in os.listdir(args.subjects_root_path)
        if os.path.isdir(os.path.join(args.subjects_root_path, name))
    ]

print(f"EXTRACTING EPISODES FROM {len(subject_list)} subjects...")

if args.output_dir is None:
    output_dir = args.subjects_root_path
else:
    output_dir = args.output_dir

if args.features is not None:
    f = open(args.features)
    feature_names = f.read().splitlines()
    print(f"Using predefined feature set: {feature_names}")
else:
    feature_names = None

# If episodes exist then remove and start over
if len(glob.glob(os.path.join(output_dir, "*", "episode*"))) > 0:
    response = input("Will need to overwrite existing episodes... continue? (y/n)")
    if response == "y":
        for f in glob.glob(os.path.join(output_dir, "*", "episode*")):
            try:
                os.remove(f)
            except OSError as ex:
                print(ex)
                sys.exit()
    else:
        print("Exiting..")
        sys.exit()

failed_to_read = 0
filter_by_nb_stays = 0
filter_by_nb_events = 0
completed_subjects = 0

for subject_dir in tqdm(subject_list, desc="Iterating over subjects"):
    dn = os.path.join(args.subjects_root_path, subject_dir)

    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except Exception:
        continue

    if not os.path.isdir(os.path.join(output_dir, subject_dir)):
        os.makedirs(os.path.join(output_dir, subject_dir), exist_ok=True)

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        # diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(
            os.path.join(args.subjects_root_path, subject_dir), args.mimic4_path
        )
    except Exception:
        if args.verbose:
            sys.stderr.write(
                f"Error: Reading data for subject {subject_id}. Events table likely to be missing/empty. \n"
            )
        failed_to_read += 1
        continue

    # Filter by number of stays per subject
    n_stays = stays.select("stay_id").collect().height
    if n_stays < args.min_stays:
        filter_by_nb_stays += 1

        if n_stays == 0:
            sys.stderr.write(
                f"Warning: Failed to get any stay data for: {subject_id}.\n"
            )

        continue

    episodic_data = assemble_episodic_data(stays)

    # clean events
    # TODO: Check these functions work as expected - currently only filling "__" values and converting to floats, but may want to do outlier detection here, or any unit conversion
    events = clean_events(events)

    if events.select("charttime").collect().height == 0:
        # no valid events for this subject
        if args.verbose:
            sys.stderr.write(f"Warning: No events found for subject {subject_id}")
        continue

    timeseries = convert_events_to_timeseries(events)

    # Ensure models have the same number of features
    if feature_names is not None:
        missing_cols = [
            x for x in feature_names if x not in timeseries.first().collect().columns
        ]
        timeseries = timeseries.with_columns(
            [pl.lit(None).alias(c) for c in missing_cols]
        )

    min_events = 1 if args.min_events is None else args.min_events
    max_events = 1e6 if args.max_events is None else args.max_events

    # extracting separate episodes per stay
    n = 0

    for row in stays.collect(streaming=True).iter_rows(named=True):
        stay_id = row["stay_id"]
        intime = row["intime"]
        outtime = row["outtime"]
        hadm_id = row["hadm_id"]
        admittime = row["admittime"]
        dischtime = row["dischtime"]

        # round everything to 1.dp
        episodic_data = episodic_data.with_columns(
            [pl.col("los").round(1), pl.col("los_ed").round(1)]
        )

        # note some stay_id = -1 but this file is just for linking to static variables and can be linked via hadm_id instead
        episodic_data.filter(pl.col("stay_id") == stay_id).collect(
            streaming=True
        ).write_csv(
            os.path.join(output_dir, subject_dir, f"episode{n+1}.csv"),
        )

        # get ed events during this specific ed/hosp stay (within certain window of time (optional))
        episode = get_events_in_period(timeseries, stay_id, hadm_id, intime, dischtime)

        if (
            episode.select(pl.col("charttime")).collect().height < min_events
            or episode.select(pl.col("charttime")).collect().height > max_events
        ):
            # if no data for this episode (or less than min or more than max) then continue
            # only keep stays with nb of datapoints within specific range
            filter_by_nb_events += 1
            continue

        # TODO: Move this to data loading process?

        # Aggregate into time-windows e.g., every 2hr by upsampling then downsampling
        # episode = episode.collect().upsample(time_column="charttime", every='2h').lazy()
        # episode = (episode.group_by_dynamic(
        #             "charttime",
        #             every="2h"
        #         ).agg(pl.exclude('charttime')).mean())

        # # Imputating of missing values using masking (adds features) or filling
        # if args.impute_strategy is not None:
        #     if args.impute_strategy == "mask":
        #         # Add new feature column with mask for whether row is nan or not
        #         for i in episode.columns:
        #             episode[i + "_isna"] = episode.loc[:, i].isna()

        #     elif args.impute_strategy == "ffill":
        #         # Fill missing values using forward fill
        #         episode = episode.ffill()
        #         episode = episode.bfill()
        #         # for remaining null values use -999
        #         episode = episode.fillna(-999)

        #     elif args.impute_strategy == "mean":
        #         mean_map = get_pop_means(args.subjects_root_path)
        #         episode = episode.fillna(mean_map)

        #     else:
        #         raise ValueError(
        #             "impute_strategy must be one of [None, mask, ffill, mean]"
        #         )

        # set index of episode as the time elapsed since ED intime
        episode = add_hours_elapsed_to_events(episode, intime).sort(by="hours")

        episode = episode.select(["hours"] + feature_names)
        episode.collect().write_csv(
            os.path.join(output_dir, subject_dir, f"episode{n+1}_timeseries.csv"),
        )

        # add to counter once data has been written to disk
        n += 1

    completed_subjects += 1

print(f"SUCCESSFULLY EXTRACTED DATA FOR {completed_subjects} SUBJECTS. \n")
print(
    f"SKIPPING {filter_by_nb_events} EVENTS, AND {filter_by_nb_stays} STAYS, FAILED TO READ {failed_to_read} SUBJECTS"
)
