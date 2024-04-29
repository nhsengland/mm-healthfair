import argparse
import os
import sys

from data.preprocessing import (
    assemble_episodic_data,
    clean_events,
)
from data.subject import (
    add_hours_elapsed_to_events,
    convert_events_to_timeseries,
    get_events_in_period,
    read_events,
    read_stays,
)
from data.util import dataframe_from_csv
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Extract episodes from per-subject data.")
parser.add_argument(
    "subjects_root_path", type=str, help="Directory containing subject sub-directories."
)
parser.add_argument(
    "--min_stays", type=int, default=1, help="Minimum number of stays per subject."
)
parser.add_argument(
    "--min_events", type=int, default=5, help="Minimum number of events per stay."
)
parser.add_argument(
    "--max_events", type=int, default=20, help="Maximum number of events per stay."
)
parser.add_argument(
    "--subject_list",
    type=str,
    help="File containing list of subject folders to include.",
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

failed_to_read = 0
filter_by_nb_stays = 0
filter_by_nb_events = 0

for subject_dir in tqdm(subject_list, desc="Iterating over subjects"):
    dn = os.path.join(args.subjects_root_path, subject_dir)

    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except Exception:
        print(f"Cannot find subject dir for {subject_id}. Exiting...")
        sys.exit()

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        # diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except Exception:
        sys.stderr.write(
            f"Error reading data for subject: {subject_id}. Events table likely to be missing/empty. \n"
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
    events = clean_events(events)

    if events.shape[0] == 0:
        # no valid events for this subject
        print(f"No events found for subject {subject_id}")
        continue

    timeseries = convert_events_to_timeseries(events)

    # extracting separate episodes

    # counter for stays that are processed, cleaned and successfully written to disk
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

        if episode.shape[0] < args.min_events or episode.shape[0] > args.max_events:
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

        # note some stay_id = -1 but this file is just for linking to static variables and can be linked via hadm_id instead
        episodic_data.loc[episodic_data.index == stay_id].to_csv(
            os.path.join(args.subjects_root_path, subject_dir, f"episode{n+1}.csv"),
            index_label="stay_id",
        )

        columns_str = [str(x) for x in list(episode.columns)]
        columns_str = map(lambda x: "" if x == "hours" else x, columns_str)
        sorted_indices = [
            i[0] for i in sorted(enumerate(columns_str), key=lambda x: x[1])
        ]

        episode = episode[[episode.columns[i] for i in sorted_indices]]
        episode.to_csv(
            os.path.join(
                args.subjects_root_path, subject_dir, f"episode{n+1}_timeseries.csv"
            ),
            index_label="hours",
        )

        # add to counter once data has been written to disk
        n += 1

print(f"SUCCESSFULLY EXTRACTED DATA FOR {n} SUBJECTS. \n")
print(
    f"SKIPPING {filter_by_nb_events} EVENTS, AND {filter_by_nb_stays} STAYS, FAILED TO READ {failed_to_read} SUBJECTS"
)
