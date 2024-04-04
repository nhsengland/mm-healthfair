# Load and preprocess MIMIC-IV v2.2 data

import argparse
import os
from pathlib import Path

import toml
from utils.haim.MIMIC_IV_HAIM_API import (
    get_unique_available_HAIM_MIMICIV_records,
    load_mimiciv,
)

parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str)  # required positional argument
parser.add_argument("--processed_dir", type=str, default="processed/")  # optional arg
args = parser.parse_args()

core_mimiciv_path = args.root_dir
modified_data_dir = args.processed_dir

# If modified data_dir does not exist, create it
if not os.path.exists(modified_data_dir):
    Path(modified_data_dir).mkdir(parents=True)

config = toml.load("config.toml")

with_cxr = True if "cxr" in config["data"]["modalities"] else False
with_notes = True if "notes" in config["data"]["modalities"] else False

dfs = load_mimiciv(core_mimiciv_path)

if not with_cxr and not with_notes:
    (
        df_admissions,
        df_patients,
        df_transfers,
        df_diagnoses_icd,
        df_drgcodes,
        df_emar,
        df_emar_detail,
        df_hcpcsevents,
        df_labevents,
        df_microbiologyevents,
        df_poe,
        df_poe_detail,
        df_prescriptions,
        df_procedures_icd,
        df_services,
        df_d_icd_diagnoses,
        df_d_icd_procedures,
        df_d_hcpcs,
        df_d_labitems,
        df_procedureevents,
        df_outputevents,
        df_inputevents,
        df_icustays,
        df_datetimeevents,
        df_chartevents,
        df_d_items,
    ) = dfs

# Save core patient data
df_base_core = df_admissions.merge(df_patients, how="left").merge(
    df_transfers, how="left"
)
df_base_core.to_csv(os.path.join(modified_data_dir, "core.csv"), index=False)

# Get all unique subject/HospAdmission/Stay Combinations
df_haim_ids = get_unique_available_HAIM_MIMICIV_records(
    df_patients,
    df_procedureevents,
    df_outputevents,
    df_inputevents,
    df_icustays,
    df_datetimeevents,
    df_chartevents,
)

df_haim_ids["haim_id"] = df_haim_ids.index
df_haim_ids = df_haim_ids.astype("int64")

# Save processed and sorted event data
df_haim_ids.to_csv(
    os.path.join(modified_data_dir, "haim_mimiciv_key_ids.csv"), index=False
)

df_haim_ids.merge(df_chartevents, how="left").merge(df_d_items, how="left").to_csv(
    os.path.join(modified_data_dir, "chartevents.csv"), index=False
)
df_haim_ids.merge(df_procedureevents, how="left").merge(df_d_items, how="left").to_csv(
    os.path.join(modified_data_dir, "procedureevents.csv"), index=False
)
df_haim_ids.merge(df_labevents, how="left").merge(df_d_labitems, how="left").to_csv(
    os.path.join(modified_data_dir, "labevents.csv"), index=False
)

print(f"Done. Processed and cleaned data saved to {modified_data_dir}")
