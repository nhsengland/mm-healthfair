import argparse
import os
import pickle

import polars as pl

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
    type=str,
    help="Directory containing processed data, will be used to output processed pkl file.",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for random sampling. Defaults to 0."
)
args = parser.parse_args()

output_dir = args.data_dir if args.output_dir is None else args.output_dir

with open(os.path.join(output_dir, "processed_data.pkl"), "rb") as f:
    data_dict = pickle.load(f)

# Load in stays data since contains the subject ids we need
stays = pl.scan_csv(os.path.join(args.data_dir, "stays.csv"))

# Generate subject-level train test split for the stays
print(f"Generating training and test split (seed {args.seed})...")

# Get IDs from processed data
processed_stay_ids = list(data_dict.keys())

processed_stays = (
    stays.select(["hadm_id", "subject_id"])
    .filter(pl.col("hadm_id").is_in(processed_stay_ids))
    .collect()
)

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
with open(os.path.join(output_dir, "training_ids.txt"), "w") as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(train_ids)
    # Write the data to the file
    file.write(data_to_write)

with open(os.path.join(output_dir, "val_ids.txt"), "w") as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(val_ids)
    # Write the data to the file
    file.write(data_to_write)

with open(os.path.join(output_dir, "test_ids.txt"), "w") as file:
    # Join the list elements into a single string with a newline character
    data_to_write = "\n".join(test_ids)
    # Write the data to the file
    file.write(data_to_write)

print("Finished.")
