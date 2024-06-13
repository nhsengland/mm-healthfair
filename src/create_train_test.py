import argparse
import os

import polars as pl
from utils.functions import load_pickle

parser = argparse.ArgumentParser(description="Create train/val/test split.")
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing extracted stays/events data.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Directory containing processed data, will be used to save id files.",
)
parser.add_argument(
    "--suffix",
    type=str,
    help="Suffix to append to id files e.g., training_ids_xxx.txt. Useful if output_dir already has generated IDs.",
)
parser.add_argument(
    "--stratify",
    action="store_true",
    help="Whether to stratify subjects by los.",
)
parser.add_argument(
    "--thresh",
    type=int,
    default=2,
    help="Threshold for subject stratification (fractional days).",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for random sampling. Defaults to 0."
)
args = parser.parse_args()

output_dir = args.data_dir if args.output_dir is None else args.output_dir

# Load in pickled data
data_dict = load_pickle(os.path.join(output_dir, "processed_data.pkl"))

# Load in stays data since contains the subject ids we need
stays = pl.scan_csv(os.path.join(args.data_dir, "stays.csv"))

# Generate subject-level train test split for the stays
print(f"Generating training and test split (seed {args.seed})...")

# Get IDs from processed data
processed_stay_ids = list(data_dict.keys())

processed_stays = (
    stays.select(["hadm_id", "subject_id", "los"])
    .filter(pl.col("hadm_id").is_in(processed_stay_ids))
    .collect()
)

if args.stratify:
    print(f"Stratifying by los > {args.thresh}...")

    # Get all subjects with positive/negative stay(s)
    pos_subjects = (
        processed_stays.filter(pl.col("los") > args.thresh)
        .unique(subset="subject_id")
        .get_column("subject_id")
        .to_list()
    )
    neg_subjects = (
        processed_stays.filter(
            ~(pl.col("los") > args.thresh) & ~(pl.col("subject_id").is_in(pos_subjects))
        )
        .unique("subject_id")
        .get_column("subject_id")
        .to_list()
    )

    # Get maximum n based on stratified samples
    max_stratified_n = min([len(pos_subjects), len(neg_subjects)])

    # Sample balanced number of subject-stays from each group (one stay per subject)

    # positive subject-stays
    pos_ids = (
        processed_stays.filter(
            (pl.col("subject_id").is_in(pos_subjects)) & (pl.col("los") > args.thresh)
        )
        .unique(subset=["subject_id"])
        .get_column("hadm_id")
        .sample(n=max_stratified_n, seed=0)
        .to_list()
    )

    # negative subject-stays
    neg_ids = (
        processed_stays.filter(
            (pl.col("subject_id").is_in(neg_subjects)) & ~(pl.col("los") > args.thresh)
        )
        .unique(subset=["subject_id"])
        .get_column("hadm_id")
        .sample(n=max_stratified_n, seed=0)
        .to_list()
    )

    stratified_ids = pos_ids + neg_ids
    processed_stays = processed_stays.filter(pl.col("hadm_id").is_in(stratified_ids))

# Get all unique subjects in processed days
subjects = processed_stays.unique(subset="subject_id", keep="first")

# 80% subjects for training + val, 20% test subjects
dev_subjects = subjects.sample(fraction=0.8, shuffle=True, seed=args.seed).get_column(
    "subject_id"
)

dev_ids = processed_stays.filter(
    pl.col.subject_id.is_in(dev_subjects.to_list())
).get_column("hadm_id")

# Remaining stays from 20% subjects left
test_ids = processed_stays.filter(~pl.col.hadm_id.is_in(dev_ids.to_list())).get_column(
    "hadm_id"
)  # remaining stays are test set

# 10% of dev set for validation
val_subjects = dev_subjects.sample(fraction=0.1, shuffle=True, seed=args.seed)

val_ids = processed_stays.filter(
    pl.col.subject_id.is_in(val_subjects.to_list())
).get_column("hadm_id")

train_ids = processed_stays.filter(
    ~pl.col.hadm_id.is_in(val_ids.to_list() + test_ids.to_list())
).get_column("hadm_id")

print(
    f"STAYS: {len(dev_ids)+len(test_ids)}\n\tTraining stays: {len(train_ids)}\n\tVal stays: {len(val_ids)}\n\tTest stays: {len(test_ids)}.\nSUBJECTS: {len(subjects)}\n\tTraining subjects: {len(dev_subjects)-len(val_subjects)}.\n\tVal subjects: {len(val_subjects)}\n\tTest subjects: {len(subjects)-len(dev_subjects)}."
)

train_ids = train_ids.cast(pl.String).to_list()
val_ids = val_ids.cast(pl.String).to_list()
test_ids = test_ids.cast(pl.String).to_list()

# Using "with open" syntax to automatically close the file
with open(
    os.path.join(output_dir, "training_ids" + f"{args.suffix}" + ".txt"), "w"
) as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(train_ids)
    # Write the data to the file
    file.write(data_to_write)

with open(os.path.join(output_dir, "val_ids" + f"{args.suffix}" + ".txt"), "w") as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(val_ids)
    # Write the data to the file
    file.write(data_to_write)

with open(
    os.path.join(output_dir, "test_ids" + f"{args.suffix}" + ".txt"), "w"
) as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(test_ids)
    # Write the data to the file
    file.write(data_to_write)

print("Finished.")
