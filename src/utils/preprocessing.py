import json
import os

import numpy as np
import pandas as pd
import polars as pl
import spacy
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.functions import (
    contains_both_ltc_types,
    get_train_split_summary,
    read_icd_mapping,
    rename_fields,
)

###############################
# EHR data preprocessing
###############################


def preproc_icd_module(
    diagnoses: pl.DataFrame | pl.LazyFrame,
    icd_map_path: str = "../config/icd9to10.txt",
    map_code_colname: str = "diagnosis_code",
    only_icd10: bool = True,
    ltc_dict_path: str = "../outputs/icd10_codes.json",
    verbose=True,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Process a diagnoses dataset with ICD codes, mapping ICD-9 to ICD-10 and generating features for long-term conditions.
    Implementation is taken from the MIMIC-IV preprocessing pipeline provided by Gupta et al. (https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main).

    Args:
        diagnoses (pl.DataFrame | pl.LazyFrame): Diagnoses data.
        icd_map_path (str): Path to ICD-9 to ICD-10 mapping file.
        map_code_colname (str): Column name for ICD code in mapping.
        only_icd10 (bool): If True, only keep ICD-10 codes.
        ltc_dict_path (str): Path to JSON with LTC code groups.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Processed diagnoses data.
    """

    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()

    def standardize_icd(mapping, df, root=False, icd_num=9):
        """Takes an ICD9 -> ICD10 mapping table and a module dataframe;
        adds column with converted ICD10 column"""

        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return (
                    mapping.filter(pl.col(map_code_colname) == icd)
                    .select("icd10cm")
                    .to_series()[0]
                )
            except IndexError:
                # Handle case where no mapping is found for the ICD code
                return np.nan

        # Create new column with original codes as default
        col_name = "icd10_convert"
        if root:
            col_name = "root_" + col_name
        df = df.with_columns(pl.col("icd_code").alias(col_name).cast(pl.Utf8))

        # Convert ICD9 codes to ICD10 in a vectorized manner
        icd9_codes = (
            df.filter(pl.col("icd_version") == icd_num)
            .select("icd_code")
            .unique()
            .to_series()
            .to_list()
        )
        icd9_to_icd10_map = {code: icd_9to10(code) for code in icd9_codes}

        df = df.with_columns(
            pl.when(pl.col("icd_version") == icd_num)
            .then(
                pl.col("icd_code").apply(
                    lambda x: icd9_to_icd10_map.get(x, np.nan), return_dtype=pl.Utf8
                )
            )
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df = df.with_columns(
                pl.col(col_name)
                .apply(
                    lambda x: x[:3] if isinstance(x, str) else np.nan,
                    return_dtype=pl.Utf8,
                )
                .alias("root")
            )

        return df

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        diagnoses = standardize_icd(icd_map, diagnoses, root=True)
        diagnoses = diagnoses.filter(pl.col("root_icd10_convert").is_not_null())
        if verbose:
            print(
                "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
                diagnoses.select("root_icd10_convert").n_unique(),
            )
            print(
                "# unique ICD-10 codes (After clinical grouping ICD-10 codes)",
                diagnoses.select("root").n_unique(),
            )
            print("# Unique patients:  ", diagnoses.select("hadm_id").n_unique())

    diagnoses = diagnoses.select(
        ["subject_id", "hadm_id", "seq_num", "long_title", "root_icd10_convert"]
    )
    #### Create features for long-term chronic conditions
    if ltc_dict_path:
        with open(ltc_dict_path) as json_dict:
            ltc_dict = json.load(json_dict)
        ### Initialise long-term condition column
        diagnoses = diagnoses.with_columns(
            pl.lit("Undefined").alias("ltc_code").cast(pl.Utf8)
        )
        print("Applying LTC coding to diagnoses...")
        for ltc_group, codelist in tqdm(ltc_dict.items()):
            # print("Group:", ltc_group, "Codes:", codelist)
            for code in codelist:
                diagnoses = diagnoses.with_columns(
                    pl.when(pl.col("root_icd10_convert").str.starts_with(code))
                    .then(pl.lit(ltc_group))
                    .otherwise(pl.col("ltc_code"))
                    .alias("ltc_code")
                    .cast(pl.Utf8)
                )

    return diagnoses.lazy() if use_lazy else diagnoses


def get_ltc_features(
    admits_last: pl.DataFrame | pl.LazyFrame,
    diagnoses: pl.DataFrame | pl.LazyFrame,
    ltc_dict_path: str = "../outputs/icd10_codes.json",
    mm_cutoff: int = 1,
    cmm_cutoff: int = 3,
    verbose=True,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate features for long-term conditions and multimorbidity from ICD-10 diagnoses and custom LTC dictionary.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Admissions data.
        diagnoses (pl.DataFrame | pl.LazyFrame): ICD-10 Diagnoses data.
        ltc_dict_path (str): Path to JSON with LTC code groups.
        mm_cutoff (int): Threshold for multimorbidity.
        cmm_cutoff (int): Threshold for complex multimorbidity.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with long-term condition count features.
    """

    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()
    if isinstance(diagnoses, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Comorbidity history
    diag_flat = diagnoses.filter(pl.col("ltc_code") != "Undefined")
    if verbose:
        print(
            "Number of previous diagnoses recorded in historical ED metadata:",
            diagnoses.shape[0],
            diagnoses["subject_id"].n_unique(),
        )

    ### Create list for each row in ltc_code column
    diag_flat = diag_flat.groupby("subject_id").agg(
        pl.col("ltc_code").apply(set).alias("ltc_code")
    )

    ### If dict is populated generate categorical columns for each long-term condition
    if ltc_dict_path:
        with open(ltc_dict_path) as json_dict:
            ltc_dict = json.load(json_dict)
        for ltc_code, _ in ltc_dict.items():
            diag_flat = diag_flat.with_columns(
                pl.col("ltc_code")
                .apply(lambda x, ltc=ltc_code: 1 if ltc in x else 0)
                .alias(ltc_code)
            )

    ### Create features for multimorbidity
    diag_flat = diag_flat.with_columns(
        [
            pl.col("ltc_code")
            .apply(contains_both_ltc_types, return_dtype=pl.Int8)
            .alias("phys_men_multimorbidity"),
            pl.col("ltc_code")
            .apply(len, return_dtype=pl.Int8)
            .alias("n_unique_conditions"),
            pl.when(pl.col("ltc_code").apply(len, return_dtype=pl.Int8) > mm_cutoff)
            .then(1)
            .otherwise(0)
            .alias("is_multimorbid"),
            pl.when(pl.col("ltc_code").apply(len, return_dtype=pl.Int8) > cmm_cutoff)
            .then(1)
            .otherwise(0)
            .alias("is_complex_multimorbid"),
        ]
    )

    ### Merge with base patient data
    admits_last = admits_last.join(diag_flat, on="subject_id", how="left")
    admits_last = admits_last.with_columns(
        [
            pl.col(col).cast(pl.Int8).fill_null(0)
            for col in diag_flat.drop(["subject_id", "ltc_code"]).columns
        ]
    )

    return admits_last.lazy() if use_lazy else admits_last


def transform_sensitive_attributes(ed_pts: pl.DataFrame) -> pl.DataFrame:
    """
    Map sensitive attributes (race, marital status) to predefined categories and types.

    Args:
        ed_pts (pl.DataFrame): Patient data.

    Returns:
        pl.DataFrame: Updated patient data.
    """

    ed_pts = ed_pts.with_columns(
        [
            pl.col("anchor_age").cast(pl.Int16),
            pl.when(
                pl.col("race")
                .str.to_lowercase()
                .str.contains("white|middle eastern|portuguese")
            )
            .then(pl.lit("White"))
            .when(
                pl.col("race").str.to_lowercase().str.contains("black|caribbean island")
            )
            .then(pl.lit("Black"))
            .when(
                pl.col("race")
                .str.to_lowercase()
                .str.contains("hispanic|south american")
            )
            .then(pl.lit("Hispanic/Latino"))
            .when(pl.col("race").str.to_lowercase().str.contains("asian"))
            .then(pl.lit("Asian"))
            .otherwise(pl.lit("Other"))
            .alias("race_group"),
            pl.col("marital_status").str.to_lowercase().str.to_titlecase(),
        ]
    )

    return ed_pts


def prepare_medication_features(
    medications: pl.DataFrame | pl.LazyFrame,
    admits_last: pl.DataFrame | pl.LazyFrame,
    top_n: int = 50,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate count and temporal (days since prescription) features for drug-level medication history.

    Args:
        medications (pl.DataFrame | pl.LazyFrame): Medication data.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations data.
        top_n (int): Number of top medications to include.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with medication count features.
    """
    if isinstance(medications, pl.LazyFrame):
        medications = medications.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Convert to pandas for easier manipulation
    medications = medications.to_pandas()
    admits_last = admits_last.to_pandas()

    medications["charttime"] = pd.to_datetime(medications["charttime"])
    medications["edregtime"] = pd.to_datetime(medications["edregtime"])
    medications = medications[(medications["charttime"] < medications["edregtime"])]

    ### Clean and prepare medication text
    medications["medication"] = (
        medications["medication"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    ### Get top_n (most commonly found) medications
    top_meds = medications["medication"].value_counts().head(top_n).index.tolist()

    #### Filter most common medications
    medications = medications[medications["medication"].isin(top_meds)]

    ### Clean some of the top medication fields
    medications["medication"] = np.where(
        medications["medication"].str.contains("vancomycin"),
        "vancomycin",
        medications["medication"],
    )
    medications["medication"] = np.where(
        medications["medication"].str.contains("acetaminophen"),
        "acetaminophen",
        medications["medication"],
    )
    medications["medication"] = np.where(
        medications["medication"].str.contains("albuterol_0.083%_neb_soln"),
        "albuterol_neb_soln",
        medications["medication"],
    )
    medications["medication"] = np.where(
        medications["medication"].str.contains(
            "oxycodone_(immediate_release)", regex=False
        ),
        "oxycodone",
        medications["medication"],
    )
    ### Get days since first and last medication
    medications = medications.sort_values(["subject_id", "medication", "charttime"])
    meds_min = medications.drop_duplicates(
        subset=["subject_id", "medication"], keep="first"
    )
    meds_max = medications.drop_duplicates(
        subset=["subject_id", "medication"], keep="last"
    )
    meds_min = meds_min.rename(columns={"charttime": "first_date"})
    meds_max = meds_max.rename(columns={"charttime": "last_date"})

    meds_min["dsf"] = (medications["edregtime"] - meds_min["first_date"]).dt.days
    meds_max["dsl"] = (medications["edregtime"] - meds_max["last_date"]).dt.days

    ### Get number of prescriptions
    meds_ids = (
        medications.groupby(["subject_id", "medication", "edregtime"])
        .size()
        .reset_index(name="n_presc")
    )

    meds_ids = meds_ids.merge(
        meds_min[["subject_id", "medication", "dsf"]],
        on=["subject_id", "medication"],
        how="left",
    )
    meds_ids = meds_ids.merge(
        meds_max[["subject_id", "medication", "dsl"]],
        on=["subject_id", "medication"],
        how="left",
    )

    #### Pivot table and create drug-specific features
    meds_piv = meds_ids.pivot_table(
        index="subject_id",
        columns="medication",
        values=["n_presc", "dsf", "dsl"],
        fill_value=0,
    )
    meds_piv.columns = [
        rename_fields("_".join(col).strip()) for col in meds_piv.columns.values
    ]

    meds_piv_total = (
        meds_ids.groupby("subject_id")["medication"]
        .nunique()
        .reset_index(name="total_n_presc")
    )

    admits_last = admits_last.merge(meds_piv_total, on="subject_id", how="left")
    admits_last = admits_last.merge(meds_piv, on="subject_id", how="left")

    ### Fill missing values
    days_cols = [col for col in admits_last.columns if "dsf" in col or "dsl" in col]
    admits_last[days_cols] = admits_last[days_cols].fillna(9999).astype(np.int32)

    nums_cols = [col for col in admits_last.columns if "n_presc" in col]
    admits_last[nums_cols] = admits_last[nums_cols].fillna(0).astype(np.int16)

    admits_last["total_n_presc"] = (
        admits_last["total_n_presc"].fillna(0).astype(np.int8)
    )
    admits_last.columns = admits_last.columns.str.replace("(", "").str.replace(")", "")

    admits_last = pl.DataFrame(admits_last)

    return admits_last.lazy() if use_lazy else admits_last


def encode_categorical_features(ehr_data: pl.DataFrame) -> pl.DataFrame:
    """
    Apply one-hot encoding to categorical features in EHR data.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """

    # prepare attribute features for one-hot-encoding
    ehr_data = ehr_data.with_columns(
        [
            pl.when(pl.col("race_group") == "Hispanic/Latino")
            .then(pl.lit("Hispanic_Latino"))
            .otherwise(pl.col("race_group"))
            .alias("race_group"),
            pl.when(pl.col("gender") == "F")
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("gender_F"),
        ]
    )
    ehr_data = ehr_data.to_dummies(
        columns=["race_group", "marital_status", "insurance"]
    )
    ehr_data = ehr_data.drop(["race", "gender"])
    ### Drop temporal columns if only a few are retained (for MLP classifier stability)
    ehr_data = ehr_data.drop(
        [col for col in ehr_data.columns if col.startswith("dsf_")]
    )
    ehr_data = ehr_data.drop(
        [col for col in ehr_data.columns if col.startswith("dsl_")]
    )
    return ehr_data


def extract_lookup_fields(
    ehr_data: pl.DataFrame,
    lookup_list: list = None,
    lookup_output_path: str = "../outputs/reference",
) -> pl.DataFrame:
    """
    Extract date and summary fields not suitable for training into a separate DataFrame.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        lookup_list (list): List of columns to extract.
        lookup_output_path (str): Directory to save lookup fields.

    Returns:
        pl.DataFrame: EHR data with lookup fields removed.
    """
    ehr_lookup = ehr_data.select(["subject_id"] + lookup_list)
    ehr_data = ehr_data.drop(lookup_list)
    print(f"Saving lookup fields in EHR data to {lookup_output_path}")
    ehr_lookup.write_csv(os.path.join(lookup_output_path, "ehr_lookup.csv"))
    return ehr_data


def remove_correlated_features(
    ehr_data: pl.DataFrame,
    feats_to_save: list = None,
    threshold: float = 0.9,
    method: str = "pearson",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Drop highly correlated features from EHR data, keeping specified features.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        feats_to_save (list): Features to keep.
        threshold (float): Correlation threshold.
        method (str): Correlation method. Defaults to Pearson's R.
        verbose (bool): If True, print summary.

    Returns:
        pl.DataFrame: EHR data with correlated features removed.
    """
    ### Specify features to save
    ehr_save = ehr_data.select(["subject_id"] + feats_to_save)
    ehr_data = ehr_data.drop(["subject_id"] + feats_to_save)
    ### Generate a linear correlation matrix
    corr_matrix = ehr_data.to_pandas().corr(method=method)
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in tqdm(iters, desc="Dropping highly correlated features..."):
        for j in range(i + 1):
            item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
            colname = item.columns
            val = abs(item.values)
            if val >= threshold:
                drop_cols.append(colname.values[0])

    to_drop = list(set(drop_cols))
    ehr_data = ehr_data.drop(to_drop)
    ehr_data = ehr_save.select(["subject_id"]).hstack(ehr_data)
    ehr_data = ehr_data.join(ehr_save, on="subject_id", how="left")

    if verbose:
        print(f"Dropped {len(to_drop)} highly correlated features.")
        print("-------------------------------------")
        print("Full list of dropped features:", to_drop)
        print("-------------------------------------")
        print(
            f"Final number of EHR features: {ehr_data.shape[1]}/{len(to_drop) + ehr_data.shape[1]}"
        )

    return ehr_data


def generate_train_val_test_set(
    ehr_data: pl.DataFrame,
    output_path: str = "../outputs/processed_data",
    outcome_col: str = "in_hosp_death",
    output_summary_path: str = "../outputs/exp_data",
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cont_cols: list = None,
    nn_cols: list = None,
    disp_dict: dict = None,
    stratify: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Create train/val/test split from static EHR data and save patient IDs across each split.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        output_path (str): Directory to save split IDs.
        outcome_col (str): Outcome column name.
        output_summary_path (str): Directory to save summary.
        seed (int): Random seed.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        cont_cols (list): Continuous columns.
        nn_cols (list): Non-normal columns.
        disp_dict (dict): Display name mapping.
        stratify (bool): If True, stratify splits balancing the sets by outcome prevalence, gender and ethnicity.
        verbose (bool): If True, print summary.

    Returns:
        dict: Dictionary with train, val, and test DataFrames.
    """
    ### Define display dictionary
    disp_dict = {
        "anchor_age": "Age",
        "gender": "Gender",
        "race_group": "Ethnicity",
        "insurance": "Insurance",
        "marital_status": "Marital status",
        "in_hosp_death": "In-hospital death",
        "ext_stay_7": "Extended stay",
        "non_home_discharge": "Non-home discharge",
        "icu_admission": "ICU admission",
        "is_multimorbid": "Multimorbidity",
        "is_complex_multimorbid": "Complex multimorbidity",
    }
    cont_cols = ["Age"]
    ### List non-normally distributed columns here for re-scaling
    ## TODO: Would need to move this to an appropriate config file
    nn_cols = [
        "total_n_presc",
        "n_unique_conditions",
        "n_presc_acetaminophen",
        "n_presc_acyclovir",
        "n_presc_albuterol_neb_soln",
        "n_presc_amlodipine",
        "n_presc_apixaban",
        "n_presc_aspirin",
        "n_presc_atorvastatin",
        "n_presc_calcium_carbonate",
        "n_presc_carvedilol",
        "n_presc_cefepime",
        "n_presc_ceftriaxone",
        "n_presc_docusate_sodium",
        "n_presc_famotidine",
        "n_presc_folic_acid",
        "n_presc_furosemide",
        "n_presc_gabapentin",
        "n_presc_heparin",
        "n_presc_hydralazine",
        "n_presc_hydromorphone_dilaudid",
        "n_presc_insulin",
        "n_presc_ipratropium_albuterol_neb",
        "n_presc_lactulose",
        "n_presc_levetiracetam",
        "n_presc_levothyroxine_sodium",
        "n_presc_lisinopril",
        "n_presc_lorazepam",
        "n_presc_metoprolol_succinate_xl",
        "n_presc_metoprolol_tartrate",
        "n_presc_metronidazole",
        "n_presc_midodrine",
        "n_presc_morphine_sulfate",
        "n_presc_multivitamins",
        "n_presc_omeprazole",
        "n_presc_ondansetron",
        "n_presc_oxycodone",
        "n_presc_pantoprazole",
        "n_presc_piperacillin_tazobactam",
        "n_presc_polyethylene_glycol",
        "n_presc_potassium_chloride",
        "n_presc_prednisone",
        "n_presc_rifaximin",
        "n_presc_senna",
        "n_presc_sevelamer_carbonate",
        "n_presc_tacrolimus",
        "n_presc_thiamine",
        "n_presc_vancomycin",
        "n_presc_vitamin_d",
        "n_presc_warfarin",
        "pon_nutrition",
        "pon_cardiology",
        "pon_respiratory",
        "pon_neurology",
        "pon_radiology",
        "pon_tpn",
        "pon_hemodialysis",
    ]

    cat_cols = [
        "In-hospital death",
        "Extended stay",
        "Non-home discharge",
        "ICU admission",
        "Multimorbidity",
        "Complex multimorbidity",
    ]

    ### Set stratification columns to include sensitive attributes + target outcome
    ehr_data = ehr_data.to_pandas()
    if stratify:
        strat_target = pd.concat(
            [ehr_data[outcome_col], ehr_data["gender"], ehr_data["race_group"]], axis=1
        )
        split_target = ehr_data.drop([outcome_col, "gender", "race_group"], axis=1)
        ### Generate split dataframes
        train_x, test_x, train_y, test_y = train_test_split(
            split_target,
            strat_target,
            test_size=(1 - train_ratio),
            random_state=seed,
            stratify=strat_target,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            test_x,
            test_y,
            test_size=test_ratio / (test_ratio + val_ratio),
            random_state=seed,
            stratify=test_y,
        )
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            ehr_data.drop([outcome_col], axis=1),
            ehr_data[outcome_col],
            test_size=(1 - train_ratio),
            random_state=seed,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            test_x,
            test_y,
            test_size=test_ratio / (test_ratio + val_ratio),
            random_state=seed,
        )

    ### Re-scale EHR data for MLP classifier
    scaler = MinMaxScaler()
    train_x[nn_cols] = scaler.fit_transform(train_x[nn_cols])
    val_x[nn_cols] = scaler.transform(val_x[nn_cols])  # Apply transformation to val_x
    test_x[nn_cols] = scaler.transform(
        test_x[nn_cols]
    )  # Apply transformation to test_x
    train_x = pd.concat([train_x, train_y], axis=1)
    val_x = pd.concat([val_x, val_y], axis=1)
    test_x = pd.concat([test_x, test_y], axis=1)
    train_x["set"] = "train"
    val_x["set"] = "val"
    test_x["set"] = "test"
    ### Print summary statistics
    if verbose:
        print(
            f"Created split with {train_x.shape[0]}({round(train_x.shape[0] / len(ehr_data), 2) * 100}%) samples in train, {val_x.shape[0]}({round(val_x.shape[0] / len(ehr_data), 2) * 100}%) samples in validation, and {test_x.shape[0]}({round(test_x.shape[0] / len(ehr_data), 2) * 100}%) samples in test."
        )
        print("Getting summary statistics for split...")
        get_train_split_summary(
            train_x,
            val_x,
            test_x,
            outcome_col,
            output_summary_path,
            cont_cols,
            nn_cols,
            disp_dict,
            cat_cols,
            verbose=verbose,
        )
        print(f"Saving train/val/test split IDs to {output_path}")

    ### Save patient IDs
    train_x[["subject_id"]].to_csv(
        os.path.join(output_path, "training_ids_" + outcome_col + ".csv"), index=False
    )
    val_x[["subject_id"]].to_csv(
        os.path.join(output_path, "validation_ids_" + outcome_col + ".csv"), index=False
    )
    test_x[["subject_id"]].to_csv(
        os.path.join(output_path, "testing_ids_" + outcome_col + ".csv"), index=False
    )

    return {"train": train_x, "val": val_x, "test": test_x}


###############################
# Notes preprocessing
###############################


def clean_notes(notes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Clean notes data by removing special characters and extra whitespaces.

    Args:
        notes (pl.DataFrame | pl.LazyFrame): Notes data.

    Returns:
        pl.DataFrame or pl.LazyFrame: Cleaned notes data.
    """
    # Remove __ and any extra whitespaces
    notes = notes.with_columns(
        target=pl.col("target")
        .str.replace_all(r"___", " ")
        .str.replace_all(r"\s+", " ")
    )
    # notes = notes.with_columns(target=pl.col("target").str.replace_all(r"\s+", " "))
    return notes


def process_text_to_embeddings(notes: pl.DataFrame) -> dict:
    """
    Generate embeddings using the Bio+Discharge ClinicalBERT model pre-trained on MIMIC-III discharge summaries.
    The current setup uses a SpaCy tokenizer mapped to a PyTorch object for GPU support.
    Text length is limited to 128 tokens per clinical note, with included padding and truncation where appropriate.
    The pre-trained model is provided by Alsentzer et al. (https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT).

    Args:
        notes (pl.DataFrame): DataFrame containing notes data.

    Returns:
        dict: Mapping from subject_id to list of (sentence, embedding) pairs.
    """
    embeddings_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nlp = spacy.load("en_core_sci_md", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_Discharge_Summary_BERT"
    )
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT").to(
        device
    )
    # randomly downsample notes for testing
    # notes = notes.sample(fraction=0.05)
    for row in tqdm(
        notes.iter_rows(named=True),
        desc="Generating notes embeddings with ClinicalBERT...",
        total=notes.height,
    ):
        subj_id = row["subject_id"]
        text = row["target"]

        # Turn text into sentences
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Tokenize all sentences at once
        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True,
        ).to(device)

        # Generate embeddings for all sentences in a single forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        if sentence_embeddings.size > 0:
            embeddings = np.mean(sentence_embeddings, axis=0)
        else:
            embeddings = np.zeros((768,))  # Handle case with no sentences

        # Map each sentence to its embedding
        sentence_embedding_pairs = list(zip(sentences, embeddings, strict=False))

        # Store the mapping in the dictionary
        embeddings_dict[subj_id] = sentence_embedding_pairs

    return embeddings_dict


###############################
# Time-series preprocessing
###############################


def clean_labevents(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean lab events by removing non-integer values and outliers.

    Args:
        labs_data (pl.LazyFrame): Lab events data.

    Returns:
        pl.LazyFrame: Cleaned lab events.
    """
    labs_data = labs_data.with_columns(
        pl.col("label")
        .str.to_lowercase()
        .str.replace(" ", "_")
        .str.replace(",", "")
        .str.replace('"', "")
        .str.replace(" ", "_"),
        pl.col("charttime").cast(pl.Utf8).str.replace("T", " ").str.strip_chars(),
    )
    lab_events = labs_data.with_columns(
        value=pl.when(pl.col("value") == ".").then(None).otherwise(pl.col("value"))
    )
    lab_events = lab_events.with_columns(
        value=pl.when(pl.col("value").str.contains("_|<|ERROR"))
        .then(None)
        .otherwise(pl.col("value"))
        .cast(
            pl.Float64, strict=False
        )  # Attempt to cast to Float64, set invalid values to None
    )
    labs_data = labs_data.drop_nulls()

    # Remove outliers using 2 std from mean
    lab_events = lab_events.with_columns(
        mean=pl.col("value").mean().over(pl.count("label"))
    )
    lab_events = lab_events.with_columns(
        std=pl.col("value").std().over(pl.count("label"))
    )
    lab_events = lab_events.filter(
        (pl.col("value") < pl.col("mean") + pl.col("std") * 2)
        & (pl.col("value") > pl.col("mean") - pl.col("std") * 2)
    ).drop(["mean", "std"])

    return lab_events


def add_time_elapsed_to_events(
    events: pl.DataFrame, starttime: pl.Datetime, remove_charttime: bool = False
) -> pl.DataFrame:
    """
    Add a column for time elapsed since a reference start time.

    Args:
        events (pl.DataFrame): Events table.
        starttime (pl.Datetime): Reference start time.
        remove_charttime (bool): If True, remove charttime column.

    Returns:
        pl.DataFrame: Updated events table.
    """
    events = events.with_columns(
        elapsed=((pl.col("charttime") - starttime) / pl.duration(hours=1)).round(1)
    )

    # reorder columns
    if remove_charttime:
        events = events.drop("charttime")

    return events


def convert_events_to_timeseries(events: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long-form events to wide-form time-series.

    Args:
        events (pl.DataFrame): Long-form events.

    Returns:
        pl.DataFrame: Wide-form time-series.
    """

    metadata = (
        events.select(["charttime", "label", "value", "linksto"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime"], keep="last")
        .sort(by="charttime")
    )

    # get unique label, values and charttimes
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )

    # pivot into wide-form format
    timeseries = timeseries.pivot(
        index="charttime", columns="label", values="value"
    ).sort(by="charttime")

    # join any metadata remaining
    timeseries = timeseries.join(
        metadata.select(["charttime", "linksto"]), on="charttime", how="inner"
    )
    return timeseries


def generate_interval_dataset(
    ehr_static: pl.DataFrame,
    ts_data: pl.DataFrame,
    ehr_regtime: pl.DataFrame,
    vitals_freq: str = "5h",
    lab_freq: str = "1h",
    min_events: int = None,
    max_events: int = None,
    impute: str = "value",
    include_dyn_mean: bool = False,
    no_resample: bool = False,
    standardize: bool = False,
    max_elapsed: int = None,
    vitals_lkup: list = None,
    outcomes: list = None,
    verbose: bool = True,
) -> dict:
    """
    Generate a time-series dataset with set intervals for each event source (vital signs and lab measurements).

    Args:
        ehr_static (pl.DataFrame): Static EHR data.
        ts_data (pl.DataFrame): Time-series data.
        ehr_regtime (pl.DataFrame): Lookup dataframe for ED arrival times.
        vitals_freq (str): Frequency for vitals resampling.
        lab_freq (str): Frequency for labs resampling.
        min_events (int): Include only patients with a minimum number of events.
        max_events (int): Include only patients with a maximum number of events.
        impute (str): Imputation method. Options are "value" (filling with -1), "forward" filling, "backward" filling or "mask" creating a string indicator for missingness.
        include_dyn_mean (bool): If True, add dynamic mean features to static dataset.
        no_resample (bool): If True, skip resampling.
        standardize (bool): If True, standardize data using min-max scaling.
        max_elapsed (int): Restrict collected measurements within the set hours from ED arrival.
        vitals_lkup (list): List of vital sign features.
        outcomes (list): List of outcome columns.
        verbose (bool): If True, print summary.

    Returns:
        dict: Data dictionary and column dictionary.
    """

    data_dict = {}
    col_dict = {}
    n = 0
    filter_by_nb_events = 0
    missing_event_src = 0
    filter_by_elapsed_time = 0
    n_src = ts_data.n_unique("linksto")
    ehr_lkup = ehr_static.drop("subject_id")
    ehr_lkup = ehr_lkup.drop(outcomes)
    col_dict["static_cols"] = ehr_lkup.columns
    col_dict["dynamic0_cols"] = vitals_lkup
    col_dict["notes_cols"] = ["sentence", "embedding"]

    feature_map, freq = _prepare_feature_map_and_freq(ts_data, vitals_freq, lab_freq)
    min_events = 1 if min_events is None else int(min_events)
    max_events = 1e6 if max_events is None else int(max_events)

    ## Standardize vital signs data between 0 and 1
    if standardize:
        ts_data = _standardize_data(ts_data)
    ts_data = ts_data.sort(by=["subject_id", "charttime"])
    ehr_regtime = ehr_regtime.sort(by=["subject_id", "edregtime"])

    for id_val in tqdm(
        ts_data.unique("subject_id").get_column("subject_id").to_list(),
        desc="Generating patient-level data...",
    ):
        pt_events = ts_data.filter(pl.col("subject_id") == id_val)
        edregtime = (
            ehr_regtime.filter(pl.col("subject_id") == id_val)
            .select("edregtime")
            .head(1)
            .item()
        )
        ehr_sel = ehr_static.filter(pl.col("subject_id") == id_val)

        if pt_events.n_unique("linksto") < n_src:
            missing_event_src += 1
            continue

        write_data, ts_data_list, s_ec, s_et = _process_patient_events(
            pt_events,
            feature_map,
            freq,
            ehr_static,
            edregtime,
            min_events,
            max_events,
            impute,
            include_dyn_mean,
            no_resample,
            max_elapsed,
        )

        if s_ec:
            filter_by_nb_events += 1
            continue

        if s_et:
            filter_by_elapsed_time += 1
            continue

        if write_data:
            ## Encode count features for training
            ehr_cur = ehr_sel.drop(outcomes)
            ehr_cur = ehr_cur.drop("subject_id").to_numpy()
            data_dict[id_val] = {"static": ehr_cur}
            for outcome in outcomes:
                data_dict[id_val][outcome] = (
                    ehr_sel.select(outcome).cast(pl.Int8).to_numpy()
                )
            for _, ts in enumerate(ts_data_list):
                key = "dynamic_0" if ts.columns == vitals_lkup else "dynamic_1"
                if key == "dynamic_1" and "dynamic1_cols" not in col_dict:
                    col_dict["dynamic1_cols"] = ts.columns
                data_dict[id_val][key] = ts.to_numpy()
            n += 1

    if verbose:
        _print_summary(
            n, filter_by_nb_events, missing_event_src, filter_by_elapsed_time
        )

    return data_dict, col_dict


def _prepare_feature_map_and_freq(
    ts_data: pl.DataFrame, vitals_freq: str = "5h", lab_freq: str = "1h"
) -> tuple[dict, dict]:
    """
    Prepare a mapping of feature names and frequency for each time-series source.

    Args:
        ts_data (pl.DataFrame): Time-series data containing a 'linksto' column.
        vitals_freq (str): Frequency for vital signs.
        lab_freq (str): Frequency for lab measurements.

    Returns:
        tuple: (feature_map, freq) where feature_map is a dict mapping data source to features,
               and freq is a dict mapping data source to frequency string.
    """
    feature_map: dict = {}
    freq: dict = {}
    for src in tqdm(ts_data.unique("linksto").get_column("linksto").to_list()):
        feature_map[src] = sorted(
            ts_data.filter(pl.col("linksto") == src)
            .unique("label")
            .get_column("label")
            .to_list()
        )
        freq[src] = vitals_freq if src == "vitalsign" else lab_freq
    return feature_map, freq


def _process_patient_events(
    pt_events: pl.DataFrame,
    feature_map: dict,
    freq: dict,
    ehr_static: pl.DataFrame,
    edregtime: pl.Datetime,
    min_events: int = 1,
    max_events: int = None,
    impute: str = "value",
    include_dyn_mean: bool = False,
    no_resample: bool = False,
    max_elapsed: int = None,
) -> tuple[bool, list[pl.DataFrame]]:
    """
    Process time-series events for a single patient, handling missing features, imputation, resampling, and filtering.

    Args:
        pt_events (pl.DataFrame): Patient's time-series events.
        feature_map (dict): Mapping from source to feature names.
        freq (dict): Mapping from source to frequency string.
        ehr_static (pl.DataFrame): Static EHR data for the patient.
        edregtime (pl.Datetime): Lookup dataframe for ED registration time.
        min_events (int): Minimum number of measurements required.
        max_events (int): Maximum number of measurements required.
        impute (str): Imputation method. Options are "value" (filling with -1), "forward" filling, "backward" filling or "mask" creating a string indicator for missingness.
        include_dyn_mean (bool): If True, add dynamic mean features.
        no_resample (bool): If True, skip resampling.
        max_elapsed (int): Restrict collected measurements within the set hours from ED arrival.

    Returns:
        tuple: (write_data, ts_data_list, skipped_due_to_event_count, skipped_due_to_elapsed_time)
    """
    write_data = True
    ts_data_list = []
    skipped_due_to_event_count = False
    skipped_due_to_elapsed_time = False

    for events_by_src in pt_events.partition_by("linksto"):
        src = events_by_src.select(pl.first("linksto")).item()
        timeseries = convert_events_to_timeseries(events_by_src)

        if not _validate_event_count(timeseries, min_events, max_events):
            skipped_due_to_event_count = True
            return False, [], skipped_due_to_event_count, False

        timeseries = _handle_missing_features(timeseries, feature_map[src])
        timeseries, ehr_static = _impute_missing_values(timeseries, ehr_static, impute)

        if include_dyn_mean:
            ehr_static = _add_dynamic_mean(timeseries, ehr_static)

        if not no_resample:
            timeseries = _resample_timeseries(timeseries, freq[src])

        if max_elapsed is not None:
            timeseries = add_time_elapsed_to_events(timeseries, edregtime)
            if timeseries.filter(pl.col("elapsed") <= max_elapsed).shape[0] == 0:
                skipped_due_to_elapsed_time = True
                return False, [], False, skipped_due_to_elapsed_time

        ts_data_list.append(timeseries.select(feature_map[src]))

    return (
        write_data,
        ts_data_list,
        skipped_due_to_event_count,
        skipped_due_to_elapsed_time,
    )


def _validate_event_count(
    timeseries: pl.DataFrame, min_events: int = 1, max_events: int = 1e6
) -> bool:
    """
    Check if the number of events in the timeseries is within the specified range.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        min_events (int): Minimum number of events.
        max_events (int): Maximum number of events.

    Returns:
        bool: True if within range, False otherwise.
    """
    return min_events <= timeseries.shape[0] <= max_events


def _handle_missing_features(
    timeseries: pl.DataFrame, features: list[str] = None
) -> pl.DataFrame:
    """
    Add missing columns to the timeseries DataFrame as nulls.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        features (list): List of required feature names.

    Returns:
        pl.DataFrame: Time-series data with missing columns added as nulls.
    """
    missing_cols = [x for x in features if x not in timeseries.columns]
    return timeseries.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing_cols]
    )


def _impute_missing_values(
    timeseries: pl.DataFrame, ehr_static: pl.DataFrame, impute: str = "value"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Impute missing values in time-series and static EHR data.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        ehr_static (pl.DataFrame): Static EHR data.
        impute (str): Imputation method ("mask", "forward", "backward", "value").

    Returns:
        tuple: (imputed_timeseries, imputed_ehr_static)
    """
    if impute == "mask":
        timeseries = timeseries.with_columns(
            [pl.col(f).is_null().alias(f + "_isna") for f in timeseries.columns]
        )
        ehr_static = ehr_static.with_columns(
            [pl.col(f).is_null().alias(f + "_isna") for f in ehr_static.columns]
        )
    elif impute in ["forward", "backward"]:
        timeseries = timeseries.fill_null(strategy=impute).fill_null(value=-1)
        ehr_static = ehr_static.fill_null(value=-1)
    elif impute == "value":
        timeseries = timeseries.fill_null(value=-1)
        ehr_static = ehr_static.fill_null(value=-1)
    return timeseries, ehr_static


def _add_dynamic_mean(
    timeseries: pl.DataFrame, ehr_static: pl.DataFrame
) -> pl.DataFrame:
    """
    Add mean of dynamic features to the static EHR data.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        ehr_static (pl.DataFrame): Static EHR data.

    Returns:
        pl.DataFrame: Static EHR data with dynamic means appended.
    """
    timeseries_mean = (
        timeseries.drop(["charttime", "linksto"]).mean().with_columns(pl.all().round(3))
    )
    return ehr_static.hstack(timeseries_mean)


def _resample_timeseries(timeseries: pl.DataFrame, freq: str = "1h") -> pl.DataFrame:
    """
    Resample the time-series data to a specified frequency.

    Args:
        timeseries (pl.DataFrame): The input time-series data.
        freq (str): The frequency for resampling (e.g., "1h").

    Returns:
        pl.DataFrame: The resampled time-series data.
    """
    timeseries = timeseries.upsample(time_column="charttime", every="1m")
    return (
        timeseries.group_by_dynamic("charttime", every=freq)
        .agg(pl.col(pl.Float64).mean())
        .fill_null(strategy="forward")
    )


def _standardize_data(ts_data: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize the 'value' column in the time-series data using min-max scaling.

    Args:
        ts_data (pl.DataFrame): The input time-series data.

    Returns:
        pl.DataFrame: Standardized time-series data.
    """
    min_val = ts_data["value"].min()
    max_val = ts_data["value"].max()
    ts_data = ts_data.with_columns(
        ((pl.col("value") - min_val) / (max_val - min_val)).alias("value")
    )

    return ts_data


def _print_summary(
    n: int = 0,
    filter_by_nb_events: int = 0,
    missing_event_src: int = 0,
    filter_by_elapsed_time: int = 0,
) -> None:
    """
    Print a summary of the time-series interval generation process.

    Args:
        n (int): Number of successfully processed patients.
        filter_by_nb_events (int): Number of patients skipped due to event count.
        missing_event_src (int): Number of patients skipped due to missing sources.
        filter_by_elapsed_time (int): Number of patients skipped due to elapsed time.

    Returns:
        None
    """
    print(f"Successfully processed time-series intervals for {n} patients.")
    print(
        f"Skipping {filter_by_nb_events} patients with less or greater number of events than specified."
    )
    print(
        f"Skipping {missing_event_src} patients due to at least one missing time-series source."
    )
    print(
        f"Skipping {filter_by_elapsed_time} patients due to no measures within elapsed time."
    )
