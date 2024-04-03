# %%

# Code adapted from: https://github.com/Wang-Yuanlong/MultimodalPred/blob/master/MIMIC_IV_HAIM_API.py

###########################################################################################################
#                                  __    __       ___       __  .___  ___.
#                                 |  |  |  |     /   \     |  | |   \/   |
#                                 |  |__|  |    /  ^  \    |  | |  \  /  |
#                                 |   __   |   /  /_\  \   |  | |  |\/|  |
#                                 |  |  |  |  /  _____  \  |  | |  |  |  |
#                                 |__|  |__| /__/     \__\ |__| |__|  |__|
#
#                               HOLISTIC ARTIFICIAL INTELLIGENCE IN MEDICINE
#
###########################################################################################################
#
# Licensed under the Apache License, Version 2.0**
# You may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the License.

# -> Authors:
#      Luis R Soenksen (<soenksen@mit.edu>),
#      Yu Ma (<midsumer@mit.edu>),
#      Cynthia Zeng (<czeng12@mit.edu>),
#      Leonard David Jean Boussioux (<leobix@mit.edu>),
#      Kimberly M Villalobos Carballo (<kimvc@mit.edu>),
#      Liangyuan Na (<lyna@mit.edu>),
#      Holly Mika Wiberg (<hwiberg@mit.edu>),
#      Michael Lingzhi Li (<mlli@mit.edu>),
#      Ignacio Fuentes (<ifuentes@mit.edu>),
#      Dimitris J Bertsimas (<dbertsim@mit.edu>),
# -> Last Update: Dec 30th, 2021
# -> Changes:
#       * Added embeddings extraction wrappers
#       * Added Code for Patient parsing towards Multi-Input AI/ML to predict value of Next Lab/X-Ray with MIMIC-IV
#       * Add Model helper functions                                                  |

## Import modules

# System

# Base
import math
import pickle
import sys

import numpy as np
import pandas as pd

# Core AI/ML
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

# Scipy
from scipy.signal import find_peaks
from tqdm import tqdm

ProgressBar().register()

"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
"""


# MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU:
    def __init__(
        self,
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
    ):
        ## HOSP
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core

        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.hcpcsevents = hcpcsevents
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.poe = poe
        self.poe_detail = poe_detail
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services

        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents


# DELTA TIME CALCULATOR FROM TWO TIMESTAMPS
def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1 - t0).total_seconds() / 3600  # Result in hrs
    except Exception:
        delta_t = math.nan

    return delta_t


# GET TIMEBOUND MIMIC-IV PATIENT RECORD BY DATABASE KEYS AND TIMESTAMPS
def get_timebound_patient_icustay(Patient_ICUstay, start_hr=None, end_hr=None):
    # Inputs:
    #   Patient_ICUstay -> Patient ICU stay structure
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any

    # %% EXAMPLE OF USE
    ## Let's select a single patient
    """
    key_subject_id = 10000032
    key_hadm_id = 29079034
    key_stay_id = 39553978
    start_hr = 0
    end_hr = 24
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    """

    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core["admittime"].values[0]
    Patient_ICUstay.emar["deltacharttime"] = Patient_ICUstay.emar.apply(
        lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
        axis=1,
    )
    Patient_ICUstay.labevents["deltacharttime"] = Patient_ICUstay.labevents.apply(
        lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
        axis=1,
    )
    Patient_ICUstay.microbiologyevents["deltacharttime"] = (
        Patient_ICUstay.microbiologyevents.apply(
            lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
            axis=1,
        )
    )
    Patient_ICUstay.outputevents["deltacharttime"] = Patient_ICUstay.outputevents.apply(
        lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
        axis=1,
    )
    Patient_ICUstay.datetimeevents["deltacharttime"] = (
        Patient_ICUstay.datetimeevents.apply(
            lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
            axis=1,
        )
    )
    Patient_ICUstay.chartevents["deltacharttime"] = Patient_ICUstay.chartevents.apply(
        lambda x: date_diff_hrs(x["charttime"], admittime) if not x.empty else None,
        axis=1,
    )

    ## --> Filter by allowable time stamps
    if start_hr is not None:
        Patient_ICUstay.emar = Patient_ICUstay.emar[
            (Patient_ICUstay.emar.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.emar.deltacharttime)
        ]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[
            (Patient_ICUstay.labevents.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.labevents.deltacharttime)
        ]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[
            (Patient_ICUstay.microbiologyevents.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)
        ]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[
            (Patient_ICUstay.outputevents.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)
        ]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[
            (Patient_ICUstay.datetimeevents.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)
        ]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[
            (Patient_ICUstay.chartevents.deltacharttime >= start_hr)
            | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)
        ]

    if end_hr is not None:
        Patient_ICUstay.emar = Patient_ICUstay.emar[
            (Patient_ICUstay.emar.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.emar.deltacharttime)
        ]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[
            (Patient_ICUstay.labevents.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.labevents.deltacharttime)
        ]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[
            (Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)
        ]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[
            (Patient_ICUstay.outputevents.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)
        ]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[
            (Patient_ICUstay.datetimeevents.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)
        ]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[
            (Patient_ICUstay.chartevents.deltacharttime <= end_hr)
            | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)
        ]

    return Patient_ICUstay


# LOAD MASTER DICTIONARY OF MIMICIV EVENTS
def load_haim_event_dictionaries(root_mimiciv_path):
    # Inputs:
    #   df_d_items -> MIMIC chartevent items dictionary
    #   df_d_labitems -> MIMIC labevent items dictionary
    #   df_d_hcpcs -> MIMIC hcpcs items dictionary
    #
    # Outputs:
    #   df_patientevents_categorylabels_dict -> Dictionary with all possible event types

    # Generate dictionary for chartevents, labevents and HCPCS
    df_patientevents_categorylabels_dict = pd.DataFrame(
        columns=["eventtype", "category", "label"]
    )

    # Load dictionaries
    df_d_items = pd.read_csv(root_mimiciv_path + "icu/d_items.csv.gz")
    df_d_labitems = pd.read_csv(root_mimiciv_path + "hosp/d_labitems.csv.gz")
    df_d_hcpcs = pd.read_csv(root_mimiciv_path + "hosp/d_hcpcs.csv.gz")

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

    return df_patientevents_categorylabels_dict


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#                            Data filtering by condition and outcome                              |
#                                                                                                 |
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API
(https://mimic-iv.mit.edu/docs/)
"""


# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR KEYWORD MATCHING
def is_haim_patient_keyword_match(patient, keywords, verbose=0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   keywords -> List of string keywords to attempt to match in an "OR" basis
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_key -> Boolean flag indicating if any of the input Keywords are present
    #   keyword_mask -> Array indicating which of the input Keywords are present (0-Absent, 1-Present)

    # Retrieve list of all the contents of patient datastructures
    patient_dfs_list = [  ## CORE
        patient.core,
        ## HOSP
        patient.diagnoses_icd,
        patient.drgcodes,
        patient.emar,
        patient.emar_detail,
        patient.hcpcsevents,
        patient.labevents,
        patient.microbiologyevents,
        patient.poe,
        patient.poe_detail,
        patient.prescriptions,
        patient.procedures_icd,
        patient.services,
        ## ICU
        patient.procedureevents,
        patient.outputevents,
        patient.inputevents,
        patient.icustays,
        patient.datetimeevents,
        patient.chartevents,
    ]

    patient_dfs_dict = [
        "core",
        "diagnoses_icd",
        "drgcodes",
        "emar",
        "emar_detail",
        "hcpcsevents",
        "labevents",
        "microbiologyevents",
        "poe",
        "poe_detail",
        "prescriptions",
        "procedures_icd",
        "services",
        "procedureevents",
        "outputevents",
        "inputevents",
        "icustays",
        "datetimeevents",
        "chartevents",
    ]

    # Initialize query mask
    keyword_mask = np.zeros([len(patient_dfs_list), len(keywords)])
    for idx_df, patient_df in enumerate(patient_dfs_list):
        for idx_keyword, keyword in enumerate(keywords):
            try:
                patient_df_text = patient_df.astype(str)
                is_df_key = (
                    patient_df_text.sum(axis=1).str.contains(keyword, case=False).any()
                )

                if is_df_key:
                    keyword_mask[idx_df, idx_keyword] = 1
                    if verbose >= 2:  # noqa: PLR2004
                        print("")
                        print(
                            "Keyword: "
                            + '"'
                            + keyword
                            + ' " '
                            + '(Found in "'
                            + patient_dfs_dict[idx_df]
                            + '" table )'
                        )
                        print(patient_df_text)
                else:
                    keyword_mask[idx_df, idx_keyword] = 0

            except Exception:
                is_df_key = False
                keyword_mask[idx_df, idx_keyword] = 0

    # Create final keyword mask
    if keyword_mask.any():
        is_key = True
    else:
        is_key = False

    return is_key, keyword_mask


# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR INCLUSION CRITERIA MATCHING
def is_haim_patient_inclusion_criteria_match(patient, inclusion_criteria, verbose=0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   inclusion_criteria -> Inclusion criteria in groups of keywords.
    #                         Keywords in groups are follow and "OR" logic,
    #                         while an "AND" logic is stablished among groups
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_included -> Boolean flag if inclusion criteria is found in patient
    #   inclusion_criteria_mask -> Binary mask of inclusion criteria found in patient

    # Clean out process bar before starting
    inclusion_criteria_mask = np.zeros(len(inclusion_criteria))
    for idx_keywords, keywords in enumerate(inclusion_criteria):
        is_included_flag, _ = is_haim_patient_keyword_match(patient, keywords, verbose)
        inclusion_criteria_mask[idx_keywords] = is_included_flag

    if inclusion_criteria_mask.all():
        is_included = True
    else:
        is_included = False

    # Print if patient has to be included
    if verbose >= 2:  # noqa: PLR2004
        print("")
        print("Inclusion Criteria: " + str(inclusion_criteria))
        print(
            "Inclusion Vector: "
            + str(inclusion_criteria_mask)
            + " , To include: "
            + str(is_included)
        )

    return is_included, inclusion_criteria_mask


# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def search_key_mimiciv_patients(
    df_haim_ids, root_mimiciv_path, inclusion_criteria, verbose=0
):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   root_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced

    # Clean out process bar before starting
    sys.stdout.flush()

    # List of key patients
    key_haim_patient_ids = []

    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total=nfiles) as pbar:
        # Update process bar
        nbase = 0
        pbar.update(nbase)
        # Iterate through all patients
        for haim_patient_idx in range(nbase, nfiles):
            # Load precomputed patient file
            filename = f"{haim_patient_idx:08d}" + ".pkl"
            patient = load_patient_object(root_mimiciv_path + "pickle/" + filename)
            # Check if patient fits keywords
            is_key, _ = is_haim_patient_inclusion_criteria_match(
                patient, inclusion_criteria, verbose
            )
            if is_key:
                key_haim_patient_ids.append(haim_patient_idx)

            # Update process bar
            pbar.update(1)

    return key_haim_patient_ids


# LOAD CORE INFO OF MIMIC IV PATIENTS
def load_core_mimic_haim_info(root_mimiciv_path, df_haim_ids):
    # Inputs:
    #   root_mimiciv_path -> Root path of mimiciv
    #   df_haim_ids -> Table of HAIM ids and corresponding keys
    #
    # Outputs:
    #   df_haim_ids_core_info -> Updated dataframe with integer representations of core data

    # %% EXAMPLE OF USE
    # df_haim_ids_core_info = load_core_mimic_haim_info(root_mimiciv_path)

    # Load core table
    df_mimiciv_core = pd.read_csv(root_mimiciv_path + "core.csv")

    # Generate integer representations of categorical variables in core
    core_var_select_list = [
        "gender",
        "ethnicity",
        "marital_status",
        "language",
        "insurance",
    ]
    core_var_select_int_list = [
        "gender_int",
        "ethnicity_int",
        "marital_status_int",
        "language_int",
        "insurance_int",
    ]
    df_mimiciv_core[core_var_select_list] = df_mimiciv_core[
        core_var_select_list
    ].astype("category")
    df_mimiciv_core[core_var_select_int_list] = df_mimiciv_core[
        core_var_select_list
    ].apply(lambda x: x.cat.codes)

    # Combine HAIM IDs with core data
    df_haim_ids_core_info = pd.merge(
        df_haim_ids, df_mimiciv_core, on=["subject_id", "hadm_id"]
    )

    return df_haim_ids_core_info


# GET DEMOGRAPHICS EMBEDDINGS OF MIMIC IV PATIENT
def get_demographic_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound mimic patient structure
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   base_embeddings -> Core base embeddings for the selected patient

    # %% EXAMPLE OF USE
    # base_embeddings = get_demographic_embeddings(dt_patient, df_haim_ids_core_info, verbose=2)

    # Retrieve dt_patient and get embeddings
    demo_embeddings = dt_patient.core.loc[
        0,
        [
            "anchor_age",
            "gender_int",
            "ethnicity_int",
            "marital_status_int",
            "language_int",
            "insurance_int",
        ],
    ]

    if verbose >= 1:
        print(demo_embeddings)

    demo_embeddings = demo_embeddings.values

    return demo_embeddings


def pivot_chartevent(df, event_list):
    # create a new table with additional columns with label list
    df1 = df[["subject_id", "hadm_id", "stay_id", "charttime"]]
    for event in event_list:
        df1[event] = np.nan
        # search in the abbreviations column
        df1.loc[(df["label"] == event), event] = df["valuenum"].astype(float)
    df_out = df1.dropna(axis=0, how="all", subset=event_list)
    return df_out


def pivot_labevent(df, event_list):
    # create a new table with additional columns with label list
    df1 = df[["subject_id", "hadm_id", "charttime"]]
    for event in event_list:
        df1[event] = np.nan
        # search in the label column
        df1.loc[(df["label"] == event), event] = df["valuenum"].astype(float)
    df_out = df1.dropna(axis=0, how="all", subset=event_list)
    return df_out


def pivot_procedureevent(df, event_list):
    # create a new table with additional columns with label list
    df1 = df[["subject_id", "hadm_id", "storetime"]]
    for event in event_list:
        df1[event] = np.nan
        # search in the label column
        df1.loc[(df["label"] == event), event] = df["value"].astype(
            float
        )  # Yu: maybe if not label use abbreviation
    df_out = df1.dropna(axis=0, how="all", subset=event_list)
    return df_out


# FUNCTION TO COMPUTE A LIST OF TIME SERIES FEATURES
def get_ts_emb(df_pivot, event_list):
    # Inputs:
    #   df_pivot -> Pivoted table
    #   event_list -> MIMIC IV Type of Event
    #
    # Outputs:
    #   df_out -> Embeddings

    # %% EXAMPLE OF USE
    # df_out = get_ts_emb(df_pivot, event_list)

    # Initialize table
    try:
        df_out = df_pivot[["subject_id", "hadm_id"]].iloc[0]
    except Exception:
        #         print(df_pivot)
        df_out = pd.DataFrame(columns=["subject_id", "hadm_id"])
    #         df_out = df_pivot[['subject_id', 'hadm_id']]

    # Adding a row of zeros to df_pivot in case there is no value
    df_pivot = df_pivot.append(pd.Series(0, index=df_pivot.columns), ignore_index=True)

    # Compute the following features
    for event in event_list:
        series = df_pivot[event].dropna()  # dropna rows
        if len(series) > 0:  # if there is any event
            df_out[event + "_max"] = series.max()
            df_out[event + "_min"] = series.min()
            df_out[event + "_mean"] = series.mean(skipna=True)
            df_out[event + "_variance"] = series.var(skipna=True)
            df_out[event + "_meandiff"] = series.diff().mean()  # average change
            df_out[event + "_meanabsdiff"] = series.diff().abs().mean()
            df_out[event + "_maxdiff"] = series.diff().abs().max()
            df_out[event + "_sumabsdiff"] = series.diff().abs().sum()
            df_out[event + "_diff"] = series.iloc[-1] - series.iloc[0]
            # Compute the n_peaks
            peaks, _ = find_peaks(series)  # , threshold=series.median()
            df_out[event + "_npeaks"] = len(peaks)
            # Compute the trend (linear slope)
            if len(series) > 1:
                df_out[event + "_trend"] = np.polyfit(
                    np.arange(len(series)), series, 1
                )[0]  # fit deg-1 poly
            else:
                df_out[event + "_trend"] = 0
    return df_out


def get_ts_embeddings(dt_patient, event_type):
    # Inputs:
    #   dt_patient -> Timebound Patient ICU stay structure
    #
    # Outputs:
    #   ts_emb -> TSfresh-like generated Lab event features for each timeseries
    #
    # %% EXAMPLE OF USE
    # ts_emb = get_labevent_ts_embeddings(dt_patient)

    # Get chartevents

    if event_type == "procedure":
        df = dt_patient.procedureevents
        # Define chart events of interest
        event_list = [
            "Foley Catheter",
            "PICC Line",
            "Intubation",
            "Peritoneal Dialysis",
            "Bronchoscopy",
            "EEG",
            "Dialysis - CRRT",
            "Dialysis Catheter",
            "Chest Tube Removed",
            "Hemodialysis",
        ]
        df_pivot = pivot_procedureevent(df, event_list)

    elif event_type == "lab":
        df = dt_patient.labevents
        # Define chart events of interest
        event_list = [
            "Glucose",
            "Potassium",
            "Sodium",
            "Chloride",
            "Creatinine",
            "Urea Nitrogen",
            "Bicarbonate",
            "Anion Gap",
            "Hemoglobin",
            "Hematocrit",
            "Magnesium",
            "Platelet Count",
            "Phosphate",
            "White Blood Cells",
            "Calcium, Total",
            "MCH",
            "Red Blood Cells",
            "MCHC",
            "MCV",
            "RDW",
            "Platelet Count",
            "Neutrophils",
            "Vancomycin",
        ]
        df_pivot = pivot_labevent(df, event_list)

    elif event_type == "chart":
        df = dt_patient.chartevents
        # Define chart events of interest
        event_list = [
            "Heart Rate",
            "Non Invasive Blood Pressure systolic",
            "Non Invasive Blood Pressure diastolic",
            "Non Invasive Blood Pressure mean",
            "Respiratory Rate",
            "O2 saturation pulseoxymetry",
            "GCS - Verbal Response",
            "GCS - Eye Opening",
            "GCS - Motor Response",
        ]
        df_pivot = pivot_chartevent(df, event_list)

    # Pivote df to record these values

    ts_emb = get_ts_emb(df_pivot, event_list)
    try:
        ts_emb = ts_emb.drop(["subject_id", "hadm_id"]).fillna(value=0)
    except Exception:
        ts_emb = (
            pd.Series(0, index=ts_emb.columns)
            .drop(["subject_id", "hadm_id"])
            .fillna(value=0)
        )

    return ts_emb


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#                     Biobert Chart Event embeddings for MIMIC-IV Deep Fusion                     |
#

"""
## -> NLP REPRESENTATION OF MIMIC-IV EHR USING TRANSFORMERS
The Transformers era originally started from the work of [(Vaswani & al., 2017)](https://arxiv.org/abs/1706.03762) who
demonstrated its superiority over [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network)
on translation tasks but it quickly extended to almost all the tasks RNNs were State-of-the-Art at that time.

One advantage of Transformer over its RNN counterpart was its non sequential attention model. Remember, the RNNs had to
iterate over each element of the input sequence one-by-one and carry an "updatable-state" between each hop. With Transformer, the model is able to look at every position in the sequence, at the same time, in one operation.

For a deep-dive into the Transformer architecture, [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks)
will drive you along all the details of the paper.

![transformer-encoder-decoder](https://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)
"""


# CONVERT SINGLE CHART EVENT DURING ICU STAY TO STRING
def chart_event_to_string(chart_event):
    # Inputs:
    #   chart_event -> Chart_event in the form of a dataframe row
    #
    # Outputs:
    #   chart_event_string -> String of chart event
    #   deltacharttime -> Time of chart event from admission

    # EXAMPLE OF USE
    # event_string, deltacharttime = chart_event_to_string(chart_event)

    deltacharttime = str(round(chart_event["deltacharttime"].values[0], 3))
    label = str(chart_event["label"].values[0])
    value = str(chart_event["value"].values[0])
    units = (
        ""
        if str(chart_event["valueuom"].values[0]) == "NaN"
        else str(chart_event["valueuom"].values[0])
    )
    warning = (
        ", Warning: outside normal" if int(chart_event["warning"].values[0]) > 0 else ""
    )
    rangeval = (
        ""
        if str(chart_event["lownormalvalue"].values[0]) == "nan"
        else (
            "range: [" + str(chart_event["lownormalvalue"].values[0]) + " - " + ""
            if str(chart_event["highnormalvalue"].values[0]) == "nan"
            else str(chart_event["highnormalvalue"].values[0]) + "]"
        )
    )

    chart_event_string = label + ": " + value + " " + units + warning + rangeval
    chart_event_string = chart_event_string.replace("nan", "").replace("NaN", "")

    return chart_event_string, deltacharttime


# CONVERT SINGLE LAB EVENT DURING ICU STAY TO STRING
def lab_event_to_string(lab_event):
    # Inputs:
    #   lab_event -> Lab_event in the form of a dataframe row
    #
    # Outputs:
    #   lab_event_string -> String of lab event
    #   deltacharttime -> Time of chart event from admission

    # %% EXAMPLE OF USE
    # lab_event_string, deltacharttime = lab_event_to_string(lab_event)

    deltacharttime = str(round(lab_event["deltacharttime"].values[0], 3))
    label = str(lab_event["label"].values[0])
    value = str(lab_event["value"].values[0])
    units = (
        ""
        if str(lab_event["valueuom"].values[0]) == "NaN"
        else str(lab_event["valueuom"].values[0])
    )
    warning = (
        ", Warning: outside normal" if not pd.isna(lab_event["flag"].values[0]) else ""
    )
    # rangeval = '' if str(lab_event['ref_range_lower'].values[0]) == 'nan' else 'range: [' + str(lab_event['ref_range_lower'].values[0]) + ' - ' + '' if str(lab_event['ref_range_upper'].values[0]) == 'nan' else str(lab_event['ref_range_upper'].values[0]) + ']'
    lab_event_string = label + ": " + value + " " + units + warning

    return lab_event_string, deltacharttime


# CONVERT SINGLE PRESCRIPTION EVENT DURING ICU STAY TO STRING
def prescription_event_to_string(prescription_event):
    # Inputs:
    #   prescription_event -> prescription_event in the form of a dataframe row
    #
    # Outputs:
    #   prescription_event_string -> String of prescription event
    #   deltacharttime -> Time of chart event from admission

    # %% EXAMPLE OF USE
    # prescription_event_string, deltacharttime = prescription_event_to_string(prescription_event)

    deltacharttime = str(round(prescription_event["deltacharttime"].values[0], 3))
    label = str(prescription_event["drug"].values[0])
    value = str(prescription_event["dose_val_rx"].values[0])
    units = (
        ""
        if str(prescription_event["dose_unit_rx"].values[0]) == "NaN"
        else str(prescription_event["dose_unit_rx"].values[0])
    )
    prescription_event_string = label + ": " + value + " " + units

    return prescription_event_string, deltacharttime


# OBTAIN LIST OF ALL EVENTS FROM CHART OF TIMEBOUND PATIENT DURING ICU STAY
def get_events_list(dt_patient, event_type, verbose):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   event_type -> Event type string
    #   verbose ->  Visualization setting for printed outputs
    #
    # Outputs:
    #   full_events_list -> List of all chart events of a single timebound patient
    #   chart_weights -> Weights of all chart events by time of occurance

    # %% EXAMPLE OF USE
    # full_events_list, event_weights = get_events_list(dt_patient, event_type, verbose)

    full_events_list = []
    event_weights = []

    if event_type == "chartevents":
        events = dt_patient.chartevents
    elif event_type == "labevents":
        events = dt_patient.labevents
    elif event_type == "prescriptions":
        events = dt_patient.prescriptions
        # Get proxi for deltachartime in prescriptions (Stop date - admition)
        admittime = dt_patient.core["admittime"][0]
        dt_patient.prescriptions["deltacharttime"] = dt_patient.prescriptions.apply(
            lambda x: date_diff_hrs(x["stoptime"], admittime) if not x.empty else None,
            axis=1,
        )

    # Sort events
    events = events.sort_values(by=["deltacharttime"])

    for idx in range(len(events)):
        # Gather chart event data
        event = events.iloc[[idx]]
        if event_type == "chartevents":
            event_string, deltacharttime = chart_event_to_string(event)
        elif event_type == "labevents":
            event_string, deltacharttime = lab_event_to_string(event)
        elif event_type == "prescriptions":
            event_string, deltacharttime = prescription_event_to_string(event)

        if verbose >= 3:  # noqa: PLR2004
            print(event_string)

        if idx == 0:
            full_events_list = [event_string]
            event_weights = [float(deltacharttime)]
        else:
            full_events_list.append(event_string)
            event_weights.append(float(deltacharttime))

    return full_events_list, event_weights


# SAVE SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def save_patient_object(obj, filepath):
    # Inputs:
    #   obj -> Timebound ICU patient stay object
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   VOID -> Object is saved in filename path
    # Overwrites any existing file.
    with open(filepath, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# LOAD SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def load_patient_object(filepath):
    # Inputs:
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   obj -> Loaded timebound ICU patient stay object

    # Overwrites any existing file.
    with open(filepath, "rb") as input:
        return pickle.load(input)


# LOAD ALL MIMIC IV TABLES IN MEMORY (warning: High memory lengthy process)
def load_mimiciv(root_mimiciv_path):  # noqa: PLR0915
    # Inputs:
    #   root_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #   filename -> Pickle filename to save object to
    #
    # Outputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables

    ### -> Initializations & Data Loading
    ###    Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)

    ## CORE
    df_admissions = dd.read_csv(
        root_mimiciv_path + "hosp/admissions.csv.gz",
        assume_missing=True,
        dtype={
            "admission_location": "object",
            "deathtime": "object",
            "edouttime": "object",
            "edregtime": "object",
        },
    )
    df_patients = dd.read_csv(
        root_mimiciv_path + "hosp/patients.csv.gz",
        assume_missing=True,
        dtype={"dod": "object"},
    )
    df_transfers = dd.read_csv(
        root_mimiciv_path + "hosp/transfers.csv.gz",
        assume_missing=True,
        dtype={"careunit": "object"},
    )

    ## HOSP
    df_d_labitems = dd.read_csv(
        root_mimiciv_path + "hosp/d_labitems.csv.gz",
        assume_missing=True,
        dtype={"loinc_code": "object"},
    )
    df_d_icd_procedures = dd.read_csv(
        root_mimiciv_path + "hosp/d_icd_procedures.csv.gz",
        assume_missing=True,
        dtype={"icd_code": "object", "icd_version": "object"},
    )
    df_d_icd_diagnoses = dd.read_csv(
        root_mimiciv_path + "hosp/d_icd_diagnoses.csv.gz",
        assume_missing=True,
        dtype={"icd_code": "object", "icd_version": "object"},
    )
    df_d_hcpcs = dd.read_csv(
        root_mimiciv_path + "hosp/d_hcpcs.csv.gz",
        assume_missing=True,
        dtype={"category": "object"},
    )
    df_diagnoses_icd = dd.read_csv(
        root_mimiciv_path + "hosp/diagnoses_icd.csv.gz",
        assume_missing=True,
        dtype={"icd_code": "object", "icd_version": "object"},
    )
    df_drgcodes = dd.read_csv(
        root_mimiciv_path + "hosp/drgcodes.csv.gz", assume_missing=True
    )
    df_emar = dd.read_csv(root_mimiciv_path + "hosp/emar.csv.gz", assume_missing=True)
    df_emar_detail = dd.read_csv(
        root_mimiciv_path + "hosp/emar_detail.csv.gz",
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
        root_mimiciv_path + "hosp/hcpcsevents.csv.gz",
        assume_missing=True,
        dtype={"hcpcs_cd": "object"},
    )
    df_labevents = dd.read_csv(
        root_mimiciv_path + "hosp/labevents.csv.gz",
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
        root_mimiciv_path + "hosp/microbiologyevents.csv.gz",
        assume_missing=True,
        dtype={"comments": "object", "quantity": "object"},
    )
    df_poe = dd.read_csv(
        root_mimiciv_path + "hosp/poe.csv.gz",
        assume_missing=True,
        dtype={
            "discontinue_of_poe_id": "object",
            "discontinued_by_poe_id": "object",
            "order_status": "object",
        },
    )
    df_poe_detail = dd.read_csv(
        root_mimiciv_path + "hosp/poe_detail.csv.gz", assume_missing=True
    )
    df_prescriptions = dd.read_csv(
        root_mimiciv_path + "hosp/prescriptions.csv.gz",
        assume_missing=True,
        dtype={"form_rx": "object", "gsn": "object"},
    )
    df_procedures_icd = dd.read_csv(
        root_mimiciv_path + "hosp/procedures_icd.csv.gz",
        assume_missing=True,
        dtype={"icd_code": "object", "icd_version": "object"},
    )
    df_services = dd.read_csv(
        root_mimiciv_path + "hosp/services.csv.gz",
        assume_missing=True,
        dtype={"prev_service": "object"},
    )

    ## ICU
    df_d_items = dd.read_csv(
        root_mimiciv_path + "icu/d_items.csv.gz", assume_missing=True
    )
    df_procedureevents = dd.read_csv(
        root_mimiciv_path + "icu/procedureevents.csv.gz",
        assume_missing=True,
        dtype={
            "value": "object",
            "secondaryordercategoryname": "object",
            "totalamountuom": "object",
        },
    )
    df_outputevents = dd.read_csv(
        root_mimiciv_path + "icu/outputevents.csv.gz",
        assume_missing=True,
        dtype={"value": "object"},
    )
    df_inputevents = dd.read_csv(
        root_mimiciv_path + "icu/inputevents.csv.gz",
        assume_missing=True,
        dtype={
            "value": "object",
            "secondaryordercategoryname": "object",
            "totalamountuom": "object",
        },
    )
    df_icustays = dd.read_csv(
        root_mimiciv_path + "icu/icustays.csv.gz", assume_missing=True
    )
    df_datetimeevents = dd.read_csv(
        root_mimiciv_path + "icu/datetimeevents.csv.gz",
        assume_missing=True,
        dtype={"value": "object"},
    )
    df_chartevents = dd.read_csv(
        root_mimiciv_path + "icu/chartevents.csv.gz",
        assume_missing=True,
        low_memory=False,
        dtype={"value": "object", "valueuom": "object"},
    )

    ### -> Data Preparation (Create full database in dask format)
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

    df_microbiologyevents["chartdate"] = dd.to_datetime(
        df_microbiologyevents["chartdate"]
    )
    df_microbiologyevents["charttime"] = dd.to_datetime(
        df_microbiologyevents["charttime"]
    )
    df_microbiologyevents["storedate"] = dd.to_datetime(
        df_microbiologyevents["storedate"]
    )
    df_microbiologyevents["storetime"] = dd.to_datetime(
        df_microbiologyevents["storetime"]
    )

    df_poe["ordertime"] = dd.to_datetime(df_poe["ordertime"])
    df_services["transfertime"] = dd.to_datetime(df_services["transfertime"])

    ## ICU
    df_procedureevents["starttime"] = dd.to_datetime(df_procedureevents["starttime"])
    df_procedureevents["endtime"] = dd.to_datetime(df_procedureevents["endtime"])
    df_procedureevents["storetime"] = dd.to_datetime(df_procedureevents["storetime"])
    df_procedureevents["comments_date"] = dd.to_datetime(
        df_procedureevents["comments_date"]
    )

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

    ### -> SORT data
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
    df_prescriptions = df_prescriptions.compute().sort_values(
        by=["subject_id", "hadm_id"]
    )
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
    df_icustays = df_icustays.compute().sort_values(
        by=["subject_id", "hadm_id", "stay_id"]
    )
    df_datetimeevents = df_datetimeevents.compute().sort_values(
        by=["subject_id", "hadm_id", "stay_id"]
    )
    df_chartevents = df_chartevents.compute().sort_values(
        by=["subject_id", "hadm_id", "stay_id"]
    )
    # --> Unwrap dictionaries
    df_d_items = df_d_items.compute()

    # Return
    return (
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
    )


# GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)
def get_unique_available_HAIM_MIMICIV_records(
    df_procedureevents,
    df_outputevents,
    df_inputevents,
    df_icustays,
    df_datetimeevents,
    df_chartevents,
):
    # Inputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables
    #
    # Outputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers

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

    df_haim_ids = df_ids

    print("Unique Subjects/Hospital Admissions/Stays Combinations: " + str(len(df_ids)))

    return df_haim_ids


# SAVE LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV
def save_unique_available_HAIM_MIMICIV_records(df_haim_ids, root_mimiciv_path):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   root_mimiciv_path -> Path to MIMIC IV Dataset
    #
    # Outputs:
    #   Saved dataframe in location

    # Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays
    df_haim_ids.to_csv(root_mimiciv_path + "haim_mimiciv_key_ids.csv", index=False)
    return print("Saved")


# GET ALL DEMOGRAPHCS DATA OF A TIMEBOUND PATIENT RECORD
def get_demographics(dt_patient):
    dem_info = dt_patient.demographics[["gender", "anchor_age", "anchor_year"]]
    dem_info["gender"] = (dem_info["gender"] == "M").astype(int)
    return dem_info.values[0]
