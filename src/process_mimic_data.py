## Code adapted from: https://github.com/lrsoenksen/HAIM/blob/main/1_Generate%20HAIM-MIMIC-MM%20from%20downloaded%20MIMIC-IV%20and%20MIMIC-CXR.ipynb

import argparse
import os
import sys

import pandas as pd
import toml
from dask import dataframe as dd
from haim.MIMIC_IV_HAIM_API import (
    Patient_ICU,
    get_timebound_patient_icustay,
    save_patient_object,
)
from tqdm import tqdm

# Import config
config = toml.load("config.toml")

parser = argparse.ArgumentParser()
parser.add_argument("root_path", type=str, help="Path to mimic root directory")
args = parser.parse_args()
root_mimiciv_path = args.root_path

# Define MIMIC IV Image Data Location (usually external drive)
# core_mimiciv_imgcxr_path = 'data/mimic-cxr-jpg/'

## CORE INFO

df_admissions = dd.read_csv(
    os.path.join(root_mimiciv_path, "hosp/admissions.csv"),
    assume_missing=True,
    dtype={
        "admission_location": "object",
        "deathtime": "object",
        "edouttime": "object",
        "edregtime": "object",
    },
)
df_patients = dd.read_csv(
    os.path.join(root_mimiciv_path, "hosp/patients.csv"),
    assume_missing=True,
    dtype={"dod": "object"},
)
df_transfers = dd.read_csv(
    root_mimiciv_path + "hosp/transfers.csv",
    assume_missing=True,
    dtype={"careunit": "object"},
)

## HOSP
df_d_labitems = dd.read_csv(
    root_mimiciv_path + "hosp/d_labitems.csv",
    assume_missing=True,
    dtype={"loinc_code": "object"},
)
df_d_icd_procedures = dd.read_csv(
    root_mimiciv_path + "hosp/d_icd_procedures.csv",
    assume_missing=True,
    dtype={"icd_code": "object", "icd_version": "object"},
)
df_d_icd_diagnoses = dd.read_csv(
    root_mimiciv_path + "hosp/d_icd_diagnoses.csv",
    assume_missing=True,
    dtype={"icd_code": "object", "icd_version": "object"},
)
df_d_hcpcs = dd.read_csv(
    root_mimiciv_path + "hosp/d_hcpcs.csv",
    assume_missing=True,
    dtype={"category": "object"},
)
df_diagnoses_icd = dd.read_csv(
    root_mimiciv_path + "hosp/diagnoses_icd.csv",
    assume_missing=True,
    dtype={"icd_code": "object", "icd_version": "object"},
)
df_drgcodes = dd.read_csv(root_mimiciv_path + "hosp/drgcodes.csv", assume_missing=True)
df_emar = dd.read_csv(
    root_mimiciv_path + "hosp/emar.csv",
    assume_missing=True,
    dtype={"enter_provider_id": "object"},
)
df_emar_detail = dd.read_csv(
    root_mimiciv_path + "hosp/emar_detail.csv",
    assume_missing=True,
    low_memory=False,
    dtype={
        "completion_interval": "object",
        "dose_due": "object",
        "dose_given": "object",
        "infusion_complete": "object",
        "infusion_rate_adjustment": "object",
        "infusion_rate_unit": "object",
        "new_iv_bag_hung": "object",
        "product_description_other": "object",
        "reason_for_no_barcode": "object",
        "restart_interval": "object",
        "route": "object",
        "side": "object",
        "site": "object",
        "continued_infusion_in_other_location": "object",
        "infusion_rate": "object",
        "non_formulary_visual_verification": "object",
        "prior_infusion_rate": "object",
        "product_amount_given": "object",
        "infusion_rate_adjustment_amount": "object",
    },
)
df_hcpcsevents = dd.read_csv(
    root_mimiciv_path + "hosp/hcpcsevents.csv",
    assume_missing=True,
    dtype={"hcpcs_cd": "object"},
)
df_labevents = dd.read_csv(
    root_mimiciv_path + "hosp/labevents.csv",
    assume_missing=True,
    dtype={
        "storetime": "object",
        "value": "object",
        "valueuom": "object",
        "flag": "object",
        "priority": "object",
        "comments": "object",
    },
)
df_microbiologyevents = dd.read_csv(
    root_mimiciv_path + "hosp/microbiologyevents.csv",
    assume_missing=True,
    dtype={
        "comments": "object",
        "quantity": "object",
        "ab_name": "object",
        "dilution_comparison": "object",
        "dilution_text": "object",
        "interpretation": "object",
        "org_name": "object",
    },
)
df_poe = dd.read_csv(
    root_mimiciv_path + "hosp/poe.csv",
    assume_missing=True,
    dtype={
        "discontinue_of_poe_id": "object",
        "discontinued_by_poe_id": "object",
        "order_status": "object",
    },
)
df_poe_detail = dd.read_csv(
    root_mimiciv_path + "hosp/poe_detail.csv", assume_missing=True
)
df_prescriptions = dd.read_csv(
    root_mimiciv_path + "hosp/prescriptions.csv",
    assume_missing=True,
    dtype={
        "form_rx": "object",
        "gsn": "object",
        "dose_val_rx": "object",
        "form_val_disp": "object",
    },
)
df_procedures_icd = dd.read_csv(
    root_mimiciv_path + "hosp/procedures_icd.csv",
    assume_missing=True,
    dtype={"icd_code": "object", "icd_version": "object"},
)
df_services = dd.read_csv(
    root_mimiciv_path + "hosp/services.csv",
    assume_missing=True,
    dtype={"prev_service": "object"},
)

## ICU
df_d_items = dd.read_csv(root_mimiciv_path + "icu/d_items.csv", assume_missing=True)
df_procedureevents = dd.read_csv(
    root_mimiciv_path + "icu/procedureevents.csv",
    assume_missing=True,
    dtype={
        "value": "object",
        "secondaryordercategoryname": "object",
        "totalamountuom": "object",
    },
)
df_outputevents = dd.read_csv(
    root_mimiciv_path + "icu/outputevents.csv",
    assume_missing=True,
    dtype={"value": "object"},
)
df_inputevents = dd.read_csv(
    root_mimiciv_path + "icu/inputevents.csv",
    assume_missing=True,
    dtype={
        "value": "object",
        "secondaryordercategoryname": "object",
        "totalamountuom": "object",
    },
)
df_icustays = dd.read_csv(root_mimiciv_path + "icu/icustays.csv", assume_missing=True)
df_datetimeevents = dd.read_csv(
    root_mimiciv_path + "icu/datetimeevents.csv",
    assume_missing=True,
    dtype={"value": "object"},
)
df_chartevents = dd.read_csv(
    root_mimiciv_path + "icu/chartevents.csv",
    assume_missing=True,
    low_memory=False,
    dtype={"value": "object", "valueuom": "object"},
)

## CXR
# df_mimic_cxr_split = dd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv', assume_missing=True)
# df_mimic_cxr_chexpert = dd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
# try:
#     df_mimic_cxr_metadata = dd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
# except Exception:
#     df_mimic_cxr_metadata = pd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
#     df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
# df_mimic_cxr_negbio = dd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-negbio.csv', assume_missing=True)

# ## NOTES
# df_noteevents = dd.from_pandas(pd.read_csv(root_mimiciv_path + 'note/noteevents.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
# df_dsnotes = dd.from_pandas(pd.read_csv(root_mimiciv_path + 'note/ds_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
# df_ecgnotes = dd.from_pandas(pd.read_csv(root_mimiciv_path + 'note/ecg_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
# df_echonotes = dd.from_pandas(pd.read_csv(root_mimiciv_path + 'note/echo_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
# df_radnotes = dd.from_pandas(pd.read_csv(root_mimiciv_path + 'note/rad_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)


# ### -> Data Preparation
# #### Create full database in dask format

# In[5]:


### Fix data type issues to allow for merging

## CORE
df_admissions["admittime"] = dd.to_datetime(df_admissions["admittime"])
df_admissions["dischtime"] = dd.to_datetime(df_admissions["dischtime"])
df_admissions["deathtime"] = dd.to_datetime(df_admissions["deathtime"])
df_admissions["edregtime"] = dd.to_datetime(df_admissions["edregtime"])
df_admissions["edouttime"] = dd.to_datetime(df_admissions["edouttime"])

df_transfers["intime"] = dd.to_datetime(df_transfers["intime"])
df_transfers["outtime"] = dd.to_datetime(df_transfers["outtime"])


## HOSP
df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.str.strip()
df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.str.strip()

df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.str.strip()

df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
df_d_hcpcs.code = df_d_hcpcs.code.str.strip()

df_prescriptions["starttime"] = dd.to_datetime(df_prescriptions["starttime"])
df_prescriptions["stoptime"] = dd.to_datetime(df_prescriptions["stoptime"])

df_emar["charttime"] = dd.to_datetime(df_emar["charttime"])
df_emar["scheduletime"] = dd.to_datetime(df_emar["scheduletime"])
df_emar["storetime"] = dd.to_datetime(df_emar["storetime"])

df_labevents["charttime"] = dd.to_datetime(df_labevents["charttime"])
df_labevents["storetime"] = dd.to_datetime(df_labevents["storetime"])

df_microbiologyevents["chartdate"] = dd.to_datetime(df_microbiologyevents["chartdate"])
df_microbiologyevents["charttime"] = dd.to_datetime(df_microbiologyevents["charttime"])
df_microbiologyevents["storedate"] = dd.to_datetime(df_microbiologyevents["storedate"])
df_microbiologyevents["storetime"] = dd.to_datetime(df_microbiologyevents["storetime"])

df_poe["ordertime"] = dd.to_datetime(df_poe["ordertime"])
df_services["transfertime"] = dd.to_datetime(df_services["transfertime"])


## ICU
df_procedureevents["starttime"] = dd.to_datetime(df_procedureevents["starttime"])
df_procedureevents["endtime"] = dd.to_datetime(df_procedureevents["endtime"])
df_procedureevents["storetime"] = dd.to_datetime(df_procedureevents["storetime"])

df_outputevents["charttime"] = dd.to_datetime(df_outputevents["charttime"])
df_outputevents["storetime"] = dd.to_datetime(df_outputevents["storetime"])

df_inputevents["starttime"] = dd.to_datetime(df_inputevents["starttime"])
df_inputevents["endtime"] = dd.to_datetime(df_inputevents["endtime"])
df_inputevents["storetime"] = dd.to_datetime(df_inputevents["storetime"])

df_icustays["intime"] = dd.to_datetime(df_icustays["intime"])
df_icustays["outtime"] = dd.to_datetime(df_icustays["outtime"])

df_datetimeevents["charttime"] = dd.to_datetime(df_datetimeevents["charttime"])
df_datetimeevents["storetime"] = dd.to_datetime(df_datetimeevents["storetime"])

df_chartevents["charttime"] = dd.to_datetime(df_chartevents["charttime"])
df_chartevents["storetime"] = dd.to_datetime(df_chartevents["storetime"])


## CXR
# df_mimic_cxr_jpg = build_mimic_cxr_jpg_dataframe(core_mimiciv_imgcxr_path + 'files', do_save=True)
# if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):
#     # Create CXRTime variable if it does not exist already
#     print("Processing CXRtime stamps")
#     df_cxr = df_mimic_cxr_metadata.compute()
#     df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
#     df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
#     df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
#     df_cxr['cxrtime'] = df_cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)

#     # # Add paths and info to images in cxr
#     # df_mimic_cxr_jpg = pd.read_csv(core_mimiciv_imgcxr_path + 'files/mimic-cxr-2.0.0-jpeg-txt.csv')
#     df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')

#     # Save
#     df_cxr.to_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', index=False)
#     #Read back the dataframe
#     try:
#         df_mimic_cxr_metadata = dd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
#     except:
#         df_mimic_cxr_metadata = pd.read_csv(root_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object', 'Note': 'object'})
#         df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
# df_mimic_cxr_metadata['cxrtime'] = dd.to_datetime(df_mimic_cxr_metadata['cxrtime'])


# ## NOTES
# df_noteevents['chartdate'] = dd.to_datetime(df_noteevents['chartdate'])
# df_noteevents['charttime'] = dd.to_datetime(df_noteevents['charttime'])
# df_noteevents['storetime'] = dd.to_datetime(df_noteevents['storetime'])

# df_dsnotes['charttime'] = dd.to_datetime(df_dsnotes['charttime'])
# df_dsnotes['storetime'] = dd.to_datetime(df_dsnotes['storetime'])

# df_ecgnotes['charttime'] = dd.to_datetime(df_ecgnotes['charttime'])
# df_ecgnotes['storetime'] = dd.to_datetime(df_ecgnotes['storetime'])

# df_echonotes['charttime'] = dd.to_datetime(df_echonotes['charttime'])
# df_echonotes['storetime'] = dd.to_datetime(df_echonotes['storetime'])

# df_radnotes['charttime'] = dd.to_datetime(df_radnotes['charttime'])
# df_radnotes['storetime'] = dd.to_datetime(df_radnotes['storetime'])

# -> SORT data
## CORE
print('PROCESSING "CORE" DB...')
df_admissions = df_admissions.compute().sort_values(by=["subject_id", "hadm_id"])
df_patients = df_patients.compute().sort_values(by=["subject_id"])
df_transfers = df_transfers.compute().sort_values(by=["subject_id", "hadm_id"])


## HOSP
print('PROCESSING "HOSP" DB...')
df_diagnoses_icd = df_diagnoses_icd.compute().sort_values(by=["subject_id"])
df_drgcodes = df_drgcodes.compute().sort_values(by=["subject_id", "hadm_id"])
df_emar = df_emar.compute().sort_values(by=["subject_id", "hadm_id"])
df_emar_detail = df_emar_detail.compute().sort_values(by=["subject_id"])
df_hcpcsevents = df_hcpcsevents.compute().sort_values(by=["subject_id", "hadm_id"])
df_labevents = df_labevents.compute().sort_values(by=["subject_id", "hadm_id"])
df_microbiologyevents = df_microbiologyevents.compute().sort_values(
    by=["subject_id", "hadm_id"]
)
df_poe = df_poe.compute().sort_values(by=["subject_id", "hadm_id"])
df_poe_detail = df_poe_detail.compute().sort_values(by=["subject_id"])
df_prescriptions = df_prescriptions.compute().sort_values(by=["subject_id", "hadm_id"])
df_procedures_icd = df_procedures_icd.compute().sort_values(
    by=["subject_id", "hadm_id"]
)
df_services = df_services.compute().sort_values(by=["subject_id", "hadm_id"])
# --> Unwrap dictionaries
df_d_icd_diagnoses = df_d_icd_diagnoses.compute()
df_d_icd_procedures = df_d_icd_procedures.compute()
df_d_hcpcs = df_d_hcpcs.compute()
df_d_labitems = df_d_labitems.compute()


## ICU
print('PROCESSING "ICU" DB...')
df_procedureevents = df_procedureevents.compute().sort_values(
    by=["subject_id", "hadm_id", "stay_id"]
)
df_outputevents = df_outputevents.compute().sort_values(
    by=["subject_id", "hadm_id", "stay_id"]
)
df_inputevents = df_inputevents.compute().sort_values(
    by=["subject_id", "hadm_id", "stay_id"]
)
df_icustays = df_icustays.compute().sort_values(by=["subject_id", "hadm_id", "stay_id"])
df_datetimeevents = df_datetimeevents.compute().sort_values(
    by=["subject_id", "hadm_id", "stay_id"]
)
df_chartevents = df_chartevents.compute().sort_values(
    by=["subject_id", "hadm_id", "stay_id"]
)
# --> Unwrap dictionaries
df_d_items = df_d_items.compute()


## CXR
# print('PROCESSING "CXR" DB...')
# df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
# df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
# df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
# df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])


# ## NOTES
# print('PROCESSING "NOTES" DB...')
# df_noteevents = df_noteevents.compute().sort_values(by=['subject_id','hadm_id'])
# df_dsnotes = df_dsnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
# df_ecgnotes = df_ecgnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
# df_echonotes = df_echonotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
# df_radnotes = df_radnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])


# In[7]:


# -> MASTER DICTIONARY of health items
# Generate dictionary for chartevents, labevents and HCPCS
df_patientevents_categorylabels_dict = pd.DataFrame(
    columns=["eventtype", "category", "label"]
)

# Get Chartevent items with labels & category
df = df_d_items
for category in sorted(df.category.astype(str).unique()):
    # print(category)
    category_list = df[df["category"] == category]
    for item in sorted(category_list.label.astype(str).unique()):
        df_patientevents_categorylabels_dict = (
            df_patientevents_categorylabels_dict.append(
                {"eventtype": "chart", "category": category, "label": item},
                ignore_index=True,
            )
        )

# Get Lab items with labels & category
df = df_d_labitems
for category in sorted(df.category.astype(str).unique()):
    # print(category)
    category_list = df[df["category"] == category]
    for item in sorted(category_list.label.astype(str).unique()):
        df_patientevents_categorylabels_dict = (
            df_patientevents_categorylabels_dict.append(
                {"eventtype": "lab", "category": category, "label": item},
                ignore_index=True,
            )
        )

# Get HCPCS items with labels & category
df = df_d_hcpcs
for category in sorted(df.category.astype(str).unique()):
    # print(category)
    category_list = df[df["category"] == category]
    for item in sorted(category_list.long_description.astype(str).unique()):
        df_patientevents_categorylabels_dict = (
            df_patientevents_categorylabels_dict.append(
                {"eventtype": "hcpcs", "category": category, "label": item},
                ignore_index=True,
            )
        )


# In[8]:


## CORE
print("- CORE > df_admissions")
print("--------------------------------")
print(df_admissions.dtypes)
print("\n\n")

print("- CORE > df_patients")
print("--------------------------------")
print(df_patients.dtypes)
print("\n\n")

print("- CORE > df_transfers")
print("--------------------------------")
print(df_transfers.dtypes)
print("\n\n")


## HOSP
print("- HOSP > df_d_labitems")
print("--------------------------------")
print(df_d_labitems.dtypes)
print("\n\n")

print("- HOSP > df_d_icd_procedures")
print("--------------------------------")
print(df_d_icd_procedures.dtypes)
print("\n\n")

print("- HOSP > df_d_icd_diagnoses")
print("--------------------------------")
print(df_d_icd_diagnoses.dtypes)
print("\n\n")

print("- HOSP > df_d_hcpcs")
print("--------------------------------")
print(df_d_hcpcs.dtypes)
print("\n\n")

print("- HOSP > df_diagnoses_icd")
print("--------------------------------")
print(df_diagnoses_icd.dtypes)
print("\n\n")

print("- HOSP > df_drgcodes")
print("--------------------------------")
print(df_drgcodes.dtypes)
print("\n\n")

print("- HOSP > df_emar")
print("--------------------------------")
print(df_emar.dtypes)
print("\n\n")

print("- HOSP > df_emar_detail")
print("--------------------------------")
print(df_emar_detail.dtypes)
print("\n\n")

print("- HOSP > df_hcpcsevents")
print("--------------------------------")
print(df_hcpcsevents.dtypes)
print("\n\n")

print("- HOSP > df_labevents")
print("--------------------------------")
print(df_labevents.dtypes)
print("\n\n")

print("- HOSP > df_microbiologyevents")
print("--------------------------------")
print(df_microbiologyevents.dtypes)
print("\n\n")

print("- HOSP > df_poe")
print("--------------------------------")
print(df_poe.dtypes)
print("\n\n")

print("- HOSP > df_poe_detail")
print("--------------------------------")
print(df_poe_detail.dtypes)
print("\n\n")

print("- HOSP > df_prescriptions")
print("--------------------------------")
print(df_prescriptions.dtypes)
print("\n\n")

print("- HOSP > df_procedures_icd")
print("--------------------------------")
print(df_procedures_icd.dtypes)
print("\n\n")

print("- HOSP > df_services")
print("--------------------------------")
print(df_services.dtypes)
print("\n\n")


## ICU
print("- ICU > df_procedureevents")
print("--------------------------------")
print(df_procedureevents.dtypes)
print("\n\n")

print("- ICU > df_outputevents")
print("--------------------------------")
print(df_outputevents.dtypes)
print("\n\n")

print("- ICU > df_inputevents")
print("--------------------------------")
print(df_inputevents.dtypes)
print("\n\n")

print("- ICU > df_icustays")
print("--------------------------------")
print(df_icustays.dtypes)
print("\n\n")

print("- ICU > df_datetimeevents")
print("--------------------------------")
print(df_datetimeevents.dtypes)
print("\n\n")

print("- ICU > df_d_items")
print("--------------------------------")
print(df_d_items.dtypes)
print("\n\n")

print("- ICU > df_chartevents")
print("--------------------------------")
print(df_chartevents.dtypes)
print("\n\n")

# ## -> GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)

df_base_core = df_admissions.merge(df_patients, how="left").merge(
    df_transfers, how="left"
)
df_base_core.to_csv(root_mimiciv_path + "core.csv")

# Get Unique Subject/HospAdmission/Stay Combinations
df_ids = pd.concat(
    [pd.DataFrame(), df_procedureevents[["subject_id", "hadm_id", "stay_id"]]],
    sort=False,
).drop_duplicates()
df_ids = pd.concat(
    [df_ids, df_outputevents[["subject_id", "hadm_id", "stay_id"]]], sort=False
).drop_duplicates()
df_ids = pd.concat(
    [df_ids, df_inputevents[["subject_id", "hadm_id", "stay_id"]]], sort=False
).drop_duplicates()
df_ids = pd.concat(
    [df_ids, df_icustays[["subject_id", "hadm_id", "stay_id"]]], sort=False
).drop_duplicates()
df_ids = pd.concat(
    [df_ids, df_datetimeevents[["subject_id", "hadm_id", "stay_id"]]], sort=False
).drop_duplicates()
df_ids = pd.concat(
    [df_ids, df_chartevents[["subject_id", "hadm_id", "stay_id"]]], sort=True
).drop_duplicates()

print("Unique Subjects: " + str(len(df_patients["subject_id"].unique())))
print("Unique Subjects/HospAdmissions/Stays Combinations: " + str(len(df_ids)))

df_haim_ids = df_ids

# Load Unique Subject/HospAdmission/Stay Combinations with Chest Xrays
df_haim_ids = pd.read_csv(root_mimiciv_path + "haim_mimiciv_key_ids.csv")
print("Unique HAIM Records Available: " + str(len(df_haim_ids)))

df_haim_ids["haim_id"] = df_haim_ids.index
df_haim_ids = df_haim_ids.astype("int64")
df_haim_ids.to_csv("cleaned_data/haim_mimiciv_key_ids.csv", index=False)

df_base_core.to_csv("cleaned_data/core.csv", index=False)
df_haim_ids.merge(df_chartevents, how="left").merge(df_d_items, how="left").to_csv(
    "cleaned_data/chartevents.csv", index=False
)
df_haim_ids.merge(df_procedureevents, how="left").merge(df_d_items, how="left").to_csv(
    "cleaned_data/procedureevents.csv", index=False
)
df_haim_ids.merge(df_labevents, how="left").merge(df_d_labitems, how="left").to_csv(
    "cleaned_data/labevents.csv", index=False
)

print("Done")


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):  # noqa: PLR0915
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    # -> FILTER data
    ##-> CORE
    f_df_base_core = df_base_core[
        (df_base_core.subject_id == key_subject_id)
        & (df_base_core.hadm_id == key_hadm_id)
    ]
    f_df_admissions = df_admissions[
        (df_admissions.subject_id == key_subject_id)
        & (df_admissions.hadm_id == key_hadm_id)
    ]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[
        (df_transfers.subject_id == key_subject_id)
        & (df_transfers.hadm_id == key_hadm_id)
    ]
    ###-> Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core = f_df_core.merge(f_df_admissions, how="left")
    f_df_core = f_df_core.merge(f_df_patients, how="left")
    f_df_core = f_df_core.merge(f_df_transfers, how="left")

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[
        (df_diagnoses_icd.subject_id == key_subject_id)
    ]
    # f_df_drgcodes = df_drgcodes[
    #     (df_drgcodes.subject_id == key_subject_id)
    #     & (df_drgcodes.hadm_id == key_hadm_id)
    # ]
    f_df_emar = df_emar[
        (df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)
    ]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[
        (df_hcpcsevents.subject_id == key_subject_id)
        & (df_hcpcsevents.hadm_id == key_hadm_id)
    ]
    f_df_labevents = df_labevents[
        (df_labevents.subject_id == key_subject_id)
        & (df_labevents.hadm_id == key_hadm_id)
    ]
    f_df_microbiologyevents = df_microbiologyevents[
        (df_microbiologyevents.subject_id == key_subject_id)
        & (df_microbiologyevents.hadm_id == key_hadm_id)
    ]
    f_df_poe = df_poe[
        (df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)
    ]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[
        (df_prescriptions.subject_id == key_subject_id)
        & (df_prescriptions.hadm_id == key_hadm_id)
    ]
    f_df_procedures_icd = df_procedures_icd[
        (df_procedures_icd.subject_id == key_subject_id)
        & (df_procedures_icd.hadm_id == key_hadm_id)
    ]
    f_df_services = df_services[
        (df_services.subject_id == key_subject_id)
        & (df_services.hadm_id == key_hadm_id)
    ]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how="left")
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how="left")
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how="left")
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how="left")

    ##-> ICU
    f_df_procedureevents = df_procedureevents[
        (df_procedureevents.subject_id == key_subject_id)
        & (df_procedureevents.hadm_id == key_hadm_id)
        & (df_procedureevents.stay_id == key_stay_id)
    ]
    f_df_outputevents = df_outputevents[
        (df_outputevents.subject_id == key_subject_id)
        & (df_outputevents.hadm_id == key_hadm_id)
        & (df_outputevents.stay_id == key_stay_id)
    ]
    f_df_inputevents = df_inputevents[
        (df_inputevents.subject_id == key_subject_id)
        & (df_inputevents.hadm_id == key_hadm_id)
        & (df_inputevents.stay_id == key_stay_id)
    ]
    f_df_icustays = df_icustays[
        (df_icustays.subject_id == key_subject_id)
        & (df_icustays.hadm_id == key_hadm_id)
        & (df_icustays.stay_id == key_stay_id)
    ]
    f_df_datetimeevents = df_datetimeevents[
        (df_datetimeevents.subject_id == key_subject_id)
        & (df_datetimeevents.hadm_id == key_hadm_id)
        & (df_datetimeevents.stay_id == key_stay_id)
    ]
    f_df_chartevents = df_chartevents[
        (df_chartevents.subject_id == key_subject_id)
        & (df_chartevents.hadm_id == key_hadm_id)
        & (df_chartevents.stay_id == key_stay_id)
    ]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how="left")
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how="left")
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how="left")
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how="left")
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how="left")

    # -> Create & Populate patient structure
    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    # Create patient object and return
    Patient_ICUstay = Patient_ICU(
        admissions,
        demographics,
        transfers,
        core,
        diagnoses_icd,
        drgcodes,
        emar,
        emar_detail,
        hcpcsevents,
        labevents,
        microbiologyevents,
        poe,
        poe_detail,
        prescriptions,
        procedures_icd,
        services,
        procedureevents,
        outputevents,
        inputevents,
        icustays,
        datetimeevents,
        chartevents,
    )

    return Patient_ICUstay


# EXTRACT ALL INFO OF A SINGLE PATIENT FROM MIMIC-IV DATASET USING HAIM ID
def extract_single_patient_records_mimiciv(
    haim_patient_idx, df_haim_ids, start_hr, end_hr
):
    # Inputs:
    #   haim_patient_idx -> Ordered number of HAIM patient
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    # Outputs:
    #   key_subject_id -> MIMIC-IV Subject ID of selected patient
    #   key_hadm_id -> MIMIC-IV Hospital Admission ID of selected patient
    #   key_stay_id -> MIMIC-IV ICU Stay ID of selected patient
    #   patient -> Full ICU patient ICU stay structure
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any

    # Extract information for patient
    key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
    key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
    key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr, end_hr)

    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient


# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def generate_all_mimiciv_patient_object(df_haim_ids, root_mimiciv_path):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   root_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced

    # Clean out
    sys.stdout.flush()

    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total=nfiles) as pbar:
        # Update process bar
        nbase = 0
        pbar.update(nbase)
        # Iterate through all patients
        for haim_patient_idx in range(nbase, nfiles):
            # Let's select each single patient and extract patient object
            start_hr = None  # Select timestamps
            end_hr = None  # Select timestamps
            key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = (
                extract_single_patient_records_mimiciv(
                    haim_patient_idx, df_haim_ids, start_hr, end_hr
                )
            )

            # Save
            filename = f"{haim_patient_idx:08d}" + ".pkl"
            save_patient_object(dt_patient, root_mimiciv_path + "pickle/" + filename)
            # Update process bar
            pbar.update(1)
    return nfiles


# In[62]:


# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
# nfiles = generate_all_mimiciv_patient_object(df_haim_ids, root_mimiciv_path)


# ## -> CHECK EVERYTHING WAS EXTRACTED CORRECTLY BY TESTING A SINGLE PATIENT RETRIEVAL AND ANALYSIS FROM HAIM-MIMIC-MM

# In[ ]:


# # Let's select a single HAIM Patient from pickle files and check if it fits inclusion criteria
# haim_patient_idx = 0

# # Select allowed timestamp range
# start_hr = None
# end_hr = None

# #Load precomputed file
# filename = f"{haim_patient_idx:08d}" + '.pkl'
# patient = load_patient_object(root_mimiciv_path + 'pickle/' + filename)
# dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)

# # Define inclusion criteria
# inclusion_criteria =[['ischemic heart disease', 'heart disease (ischemic)', 'heart disease'], ['acute respiratory failure', 'respiratory failure'], ['hypertension'],["died"]]
# is_included, inclusion_criteria_mask = is_haim_patient_inclusion_criteria_match(dt_patient, inclusion_criteria, verbose=0)
# # get_visioin_embedding(dt_patient)
