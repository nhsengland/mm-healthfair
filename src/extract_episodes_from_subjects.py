import argparse
import glob
import os
import sys

import polars as pl
from tqdm import tqdm
from utils.functions import dataframe_from_csv, get_pop_means
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
    "--impute_strategy",
    "-i",
    type=str,
    help="Which strategy to use to impute missing values in timeseries data.",
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

args = parser.parse_args()

if args.subject_list is not None:
    subject_list = (
        dataframe_from_csv(args.subject_list, header=None)[0].astype(str).to_list()
    )
else:
    subject_list = os.listdir(args.subjects_root_path)

if args.verbose:
    print(
        f"EXTRACTING EPISODES FROM {len(os.listdir(args.subjects_root_path))} subjects..."
    )

if args.output_dir is None:
    output_dir = args.subjects_root_path
else:
    output_dir = args.output_dir

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


def get_feature_list():
    features = (
        pl.concat(
            [
                pl.scan_csv(f).select(pl.col("label"))
                for f in glob.glob(
                    os.path.join(args.subjects_root_path, "*", "events.csv")
                )
            ],
            how="diagonal",
        )
        .unique()
        .collect()
    )
    return features.get_column("label").to_list()


feature_names = get_feature_list()

for subject_dir in tqdm(subject_list, desc="Iterating over subjects"):
    dn = os.path.join(args.subjects_root_path, subject_dir)

    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except Exception:
        continue

    if not os.path.isdir(os.path.join(output_dir, subject_dir)):
        os.mkdir(os.path.join(output_dir, subject_dir), parents=True, exist_ok=True)

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
    if stays.shape[0] < args.min_stays:
        filter_by_nb_stays += 1
        continue

    episodic_data = assemble_episodic_data(stays)

    if episodic_data.shape[0] == 0:
        sys.stderr.write(f"Warning: Failed to get stay data for: {subject_id}.\n")

    # clean events
    # TODO: Check these functions work as expected - currently only filling "__" values and converting to floats, but may want to do outlier detection here, or any unit conversion
    events = clean_events(events)

    if events.shape[0] == 0:
        # no valid events for this subject
        if args.verbose:
            sys.stderr.write(f"Warning: No events found for subject {subject_id}")
        continue

    timeseries = convert_events_to_timeseries(events)

    # Ensure timeseries have same number of columns (hours + features)
    timeseries = timeseries.reindex(
        columns=list(set().union(feature_names, timeseries.columns))
    )

    min_events = 1 if args.min_events is None else args.min_events
    max_events = 1e6 if args.max_events is None else args.max_events

    # extracting separate episodes per stay
    n = 0
    for stay_idx in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[stay_idx]
        intime = stays.intime.iloc[stay_idx]
        outtime = stays.outtime.iloc[stay_idx]
        hadm_id = stays.hadm_id.iloc[stay_idx]
        admittime = stays.admittime.iloc[stay_idx]
        dischtime = stays.dischtime.iloc[stay_idx]

        # get ed events during this specific ed/hosp stay (within certain window of time (optional))
        episode = get_events_in_period(timeseries, stay_id, hadm_id, intime, dischtime)

        # Aggregate into time-windows e.g., every 30mins from 00:00:00 to 23:59:59 (so all data is in same intervals)
        episode = episode.resample("2h", on="charttime").mean().reset_index()

        # Imputating of missing values using masking (adds features) or filling
        if args.impute_strategy is not None:
            if args.impute_strategy == "mask":
                # Add new feature column with mask for whether row is nan or not
                for i in episode.columns:
                    episode[i + "_isna"] = episode.loc[:, i].isna()

            elif args.impute_strategy == "ffill":
                # Fill missing values using forward fill
                episode = episode.ffill()
                episode = episode.bfill()
                # for remaining null values use -999
                episode = episode.fillna(-999)

            elif args.impute_strategy == "mean":
                mean_map = get_pop_means(args.subjects_root_path)
                episode = episode.fillna(mean_map)

            else:
                raise ValueError(
                    "impute_strategy must be one of [None, mask, ffill, mean]"
                )

        if episode.shape[0] < min_events or episode.shape[0] > max_events:
            # if no data for this episode (or less than min or more than max) then continue
            # only keep stays with nb of datapoints within specific range
            filter_by_nb_events += 1
            continue

        # set index of episode as the time elapsed since ED intime
        episode = (
            add_hours_elapsed_to_events(episode, intime)
            .set_index("hours")
            .sort_index(axis=0)
        )

        # Drop rows with negative "hours"
        episode = episode[episode.index > 0]

        # round everything to 1.dp
        episode = episode.round(1)

        # note some stay_id = -1 but this file is just for linking to static variables and can be linked via hadm_id instead
        episodic_data.loc[episodic_data.index == stay_id].to_csv(
            os.path.join(output_dir, subject_dir, f"episode{n+1}.csv"),
            index_label="stay_id",
        )

        columns_str = [str(x) for x in list(episode.columns)]
        columns_str = map(lambda x: "" if x == "hours" else x, columns_str)
        sorted_indices = [
            i[0] for i in sorted(enumerate(columns_str), key=lambda x: x[1])
        ]

        episode = episode[[episode.columns[i] for i in sorted_indices]]

        episode.to_csv(
            os.path.join(output_dir, subject_dir, f"episode{n+1}_timeseries.csv"),
            index_label="hours",
        )

        # add to counter once data has been written to disk
        n += 1

    completed_subjects += 1

print(f"SUCCESSFULLY EXTRACTED DATA FOR {completed_subjects} SUBJECTS. \n")
print(
    f"SKIPPING {filter_by_nb_events} EVENTS, AND {filter_by_nb_stays} STAYS, FAILED TO READ {failed_to_read} SUBJECTS"
)
