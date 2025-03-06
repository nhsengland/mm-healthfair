import pickle
from typing import Any

import polars as pl


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


def impute_from_df(
    impute_to: pl.DataFrame | pl.LazyFrame,
    impute_from: pl.DataFrame,
    use_col: str = None,
    key_col: str = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Imputes values from one dataframe to another.

    Args:
        impute_to (pl.DataFrame | pl.LazyFrame): Table to impute values in to.
        impute_from (pl.DataFrame): Table to impute values from.
        use_col (str, optional): Column to containing values to impute. Defaults to None.
        key_col (str, optional): Column to use to identify matching rows. Defaults to None.

    Returns:
        pl.DataFrame | pl.LazyFrame: _description_
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
    stays = stays.sort(["subject_id", "edregtime"]).unique(subset=["subject_id"], keep="last")
    return stays

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
    mapping = pl.read_csv(map_path, separator='\t', encoding='iso-8859-1')
    mapping = mapping.with_columns(pl.col("diagnosis_description").str.to_lowercase())
    return mapping

def contains_both_ltc_types(ltc_set: pl.Series) -> bool:
    """
    Helper util function for physical-mental multimorbidity detection.

    Args:
        ltc_set (pl.Series): Series containing LTC codes.

    Returns:
        bool: True if both physical and mental LTC types are present, False otherwise.
    """
    physltc_present = ltc_set.str.starts_with("physltc_").any()
    menltc_present = ltc_set.str.starts_with("menltc_").any()
    return physltc_present and menltc_present

def preview_data(filepath: str) -> None:
    """Prints a single example from data dictionary.

    Args:
        filepath (str): Path to .pkl file containing data dictionary.
    """
    data_dict = load_pickle(filepath)
    example_id = list(data_dict.keys())[-1]
    print(f"Example data:{data_dict[example_id]}")

def get_demographics_summary(ed_pts: pl.DataFrame) -> None:
    """
    Summarises sensitive attributes and outcome prevalence.
    Args:
        demographics (pl.DataFrame): Demographics data.

    Returns:
        pl.DataFrame: Summary table.
    """
    print('Demographics summary')
    print('Unique patients:', ed_pts.subject_id.nunique())
    print('Age distribution:', ed_pts.anchor_age.describe())
    print('Gender distribution:', ed_pts.gender.value_counts())
    print('-------------------------------')
    print('Health outcomes')
    print(ed_pts.in_hosp_death.value_counts(normalize=True))
    print(ed_pts.ext_stay_7.value_counts(normalize=True))
    print(ed_pts.non_home_discharge.value_counts(normalize=True))
    print(ed_pts.icu_admission.value_counts(normalize=True))
    print('-------------------------------')
    print('Comorbidity history')
    print(ed_pts.is_multimorbid.value_counts(normalize=True))
    print(ed_pts.is_complex_multimorbid.value_counts(normalize=True))

def read_from_txt(filepath: str, as_type="str") -> list:
    """Read from line-seperated txt file.

    Args:
        filepath (str): Path to text file.

    Returns:
        list: List containing data.
    """
    with open(filepath) as f:
        if as_type == "str":
            data = [str(line.strip()) for line in f.readlines()]
        elif as_type == "int":
            data = [int(line.strip()) for line in f.readlines()]
    return data
