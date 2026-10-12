import os
import pickle
from typing import Any

import pandas as pd
import polars as pl
from tableone import TableOne


def load_pickle(filepath: str) -> Any:
    """Load a pickled object.

    Args:
        filepath (str): Path to pickle (.pkl) file.

    Returns:
        Any: Loaded object.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(target: dict, filepath: str, fname: str = "mm_feat.pkl") -> Any:
    """
    Save a Python object as a pickle file.

    Args:
        target (dict): Object to pickle.
        filepath (str): Directory to save the pickle file.
        fname (str): Filename for the pickle file (default: "mm_feat.pkl").

    Returns:
        None
    """
    with open(os.path.join(filepath, fname), "wb") as f:
        pickle.dump(target, f)


def impute_from_df(
    impute_to: pl.DataFrame | pl.LazyFrame,
    impute_from: pl.DataFrame,
    use_col: str = None,
    key_col: str = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Impute values from one dataframe to another using a key column.

    Args:
        impute_to (pl.DataFrame | pl.LazyFrame): Table to impute values into.
        impute_from (pl.DataFrame): Table to impute values from.
        use_col (str, optional): Column containing values to impute.
        key_col (str, optional): Column to use to identify matching rows.

    Returns:
        pl.DataFrame | pl.LazyFrame: DataFrame with imputed values.
    """
    # create dictionary mapping values to identifier key
    dict_map = impute_from.select([key_col, use_col]).rows_by_key(
        key=use_col, unique=True
    )

    # create temporary column contaning imputed values
    impute_to = impute_to.with_columns(tmp=pl.col(use_col).replace(dict_map))

    # only use imputed values when missing in original table
    impute_to = impute_to.with_columns(
        pl.when(pl.col(key_col).is_null())
        .then(pl.col("tmp"))
        .otherwise(pl.col(key_col))
        .alias(key_col)
    ).drop("tmp")  # remove temporary column

    return impute_to


def get_final_episodes(stays: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Extracts the final ED episode with hospitalisation for creating a unique patient cohort.

    Args:
        stays (pl.DataFrame): Stays data.

    Returns:
        pl.DataFrame: Patient-level data.
    """
    if isinstance(stays, pl.LazyFrame):
        stays = stays.collect()

    ### Sort values and get final ED episode
    ### Save second to last episode for validation
    stays_final = stays.sort(["subject_id", "edregtime"]).unique(
        subset=["subject_id"], keep="last"
    )
    ### Get second-to-last episode for time-series collection
    stays_second_last = stays.sort(["subject_id", "edregtime"]).filter(
        ~pl.col("hadm_id").is_in(stays_final["hadm_id"])
    )
    stays_second_last = stays_second_last.sort(["subject_id", "edregtime"]).unique(
        subset=["subject_id"], keep="last"
    )
    stays_second_last = stays_second_last.rename(
        {"edregtime": "prev_edregtime", "dischtime": "prev_dischtime"}
    ).select(["subject_id", "prev_edregtime", "prev_dischtime"])
    stays_final = stays_final.join(stays_second_last, on="subject_id", how="left")
    ## Drop NAs in prev_dischtime or prev_edregtime
    stays_final = stays_final.filter(
        ~pl.col("prev_dischtime").is_null() & ~pl.col("prev_edregtime").is_null()
    )
    ## Drop any overlapping episodes
    stays_final = stays_final.filter(pl.col("edregtime") > pl.col("prev_dischtime"))

    return stays_final


def get_n_unique_values(
    table: pl.DataFrame | pl.LazyFrame, use_col: str = "subject_id"
) -> int:
    """Compute number of unique values in particular column in table.

    Args:
        table (pl.DataFrame | pl.LazyFrame): Table.
        use_col (str, optional): Column to use. Defaults to "subject_id".

    Returns:
        int: Number of unique values.
    """
    unique_vals = table.select(use_col).unique()
    return (
        unique_vals.count().collect(streaming=True).item()
        if type(unique_vals) == pl.LazyFrame
        else unique_vals.count().item()
    )


def scale_numeric_features(
    table: pl.DataFrame, numeric_cols: list = None, over: str = None
) -> pl.DataFrame:
    """Applies min/max scaling to numeric columns and rounds to 1 d.p.

    Args:
        table (pl.DataFrame): Table.
        numeric_cols (list, optional): List of columns to apply to. Defaults to None.
        over (str, optional): Column to group by before computing min/max. Defaults to None.

    Returns:
        pl.DataFrame: Updated table.
    """
    # Normalise/scale specified cols using MinMax scaling
    if over is None:
        scaled = table.select(
            (pl.col(numeric_cols) - pl.col(numeric_cols).min())
            / (pl.col(numeric_cols).max() - pl.col(numeric_cols).min())
        )
    else:
        # compute min max of cols over another groupby col e.g., subject_id or label
        scaled = table.select(
            (
                (pl.col(numeric_cols) - pl.col(numeric_cols).min())
                / (pl.col(numeric_cols).max() - pl.col(numeric_cols).min())
            ).over(over)
        )

    # ensure all scaled features are floats to 2 d.p
    scaled = scaled.with_columns(pl.all().round(2))
    return table.select(pl.col("*").exclude(numeric_cols)).hstack(scaled)


def read_icd_mapping(map_path: str) -> pl.DataFrame:
    """
    Reads ICD-9 to ICD-10 mapping file for chronic conditions.
    """
    mapping = pl.read_csv(map_path, separator="\t", encoding="iso-8859-1")
    mapping = mapping.with_columns(pl.col("diagnosis_description").str.to_lowercase())
    return mapping


def contains_both_ltc_types(ltc_set: set) -> bool:
    """
    Helper util function for physical-mental multimorbidity detection.

    Args:
        ltc_set (set): Set containing LTC codes.

    Returns:
        bool: True if both physical and mental LTC types are present, False otherwise.
    """
    ltc_series = pl.Series(list(ltc_set))
    physltc_present = ltc_series.str.starts_with("physltc_").any()
    menltc_present = ltc_series.str.starts_with("menltc_").any()
    return physltc_present and menltc_present


def preview_data(filepath: str) -> None:
    """Prints a single example from data dictionary.

    Args:
        filepath (str): Path to .pkl file containing data dictionary.
    """
    data_dict = load_pickle(filepath)
    example_id = list(data_dict.keys())[-1]
    print(f"Example data:{data_dict[example_id]}")


def get_demographics_summary(ed_pts: pl.DataFrame | pl.LazyFrame) -> None:
    """
    Summarises sensitive attributes and outcome prevalence.
    Args:
        demographics (pl.DataFrame): Demographics data.

    Returns:
        pl.DataFrame: Summary table.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect().to_pandas()

    print("Demographics summary")
    print("Unique patients:", ed_pts.subject_id.nunique())
    print("Age distribution:", ed_pts.anchor_age.describe())
    print("Gender distribution:", ed_pts.gender.value_counts())
    print("-------------------------------")
    print("Health outcomes")
    print(ed_pts.in_hosp_death.value_counts(normalize=True))
    print(ed_pts.ext_stay_7.value_counts(normalize=True))
    print(ed_pts.non_home_discharge.value_counts(normalize=True))
    print(ed_pts.icu_admission.value_counts(normalize=True))
    print("-------------------------------")
    print("Multimorbidity")
    print(ed_pts.is_multimorbid.value_counts(normalize=True))
    print(ed_pts.is_complex_multimorbid.value_counts(normalize=True))
    print("-------------------------------")


def get_train_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str = "in_hosp_death",
    output_path: str = "../outputs/exp_data",
    cont_cols: list = None,
    nn_cols: list = None,
    disp_dict: dict = None,
    cat_cols: list = None,
    verbose: bool = True,
) -> None:
    """
    Print and save a statistical summary for the train, validation, and test splits.

    Args:
        train (pd.DataFrame): Training set DataFrame.
        val (pd.DataFrame): Validation set DataFrame.
        test (pd.DataFrame): Test set DataFrame.
        outcome (str): Name of the outcome variable (default: "in_hosp_death").
        output_path (str): Directory to save the summary HTML file.
        cont_cols (list): List of continuous columns.
        nn_cols (list): List of non-normal columns.
        disp_dict (dict): Dictionary mapping original to display column names.
        cat_cols (list): List of categorical columns.
        verbose (bool): If True, print progress messages.

    Returns:
        None
    """
    if verbose:
        print(f"Saving demographic summary for training split of outcome {outcome}.")
    samples = pd.concat([train, val, test], axis=0)
    samples_disp = samples.rename(columns=disp_dict)
    for col in cat_cols:
        samples_disp[col] = samples_disp[col].replace({0: "N", 1: "Y"})
    sum_table = TableOne(
        samples_disp,
        [col for col in disp_dict.values()],
        order={"set": ["train", "val", "test"]},
        categorical=[col for col in disp_dict.values() if col not in cont_cols],
        overall=True,
        groupby="set",
        pval=True,
        htest_name=True,
        tukey_test=True,
        nonnormal=nn_cols,
    )
    sum_table.to_html(os.path.join(output_path, f"train_summary_{outcome}.html"))
    if verbose:
        print(f"Saved {outcome} summary to {output_path}/train_summary_{outcome}.html.")


def rename_fields(col):
    """
    Helper function to rename drug and specialty feature names.

    Args:
        col (Any): Column name or tuple of column names.

    Returns:
        str: Joined string if input is a tuple, otherwise the original column name.
    """
    if isinstance(col, tuple):
        col = "_".join(str(c) for c in col)
    return col


def read_from_txt(filepath: str, as_type="str") -> list:
    """Read from line-seperated txt file.

    Args:
        filepath (str): Path to text file.

    Returns:
        list: List containing data.
    """
    with open(filepath) as f:
        if as_type == "str":
            data = [str(line.strip()) for line in f]
        elif as_type == "int":
            data = [int(line.strip()) for line in f]
    return data
