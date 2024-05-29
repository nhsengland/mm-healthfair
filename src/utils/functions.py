import os
import pickle
from glob import glob

import polars as pl


def impute_from_df(
    impute_to: pl.DataFrame | pl.LazyFrame,
    impute_from: pl.DataFrame,
    use_col: str = None,
    key_col: str = None,
):
    dict_map = impute_from.select([key_col, use_col]).rows_by_key(
        key=use_col, unique=True
    )

    impute_to = impute_to.with_columns(tmp=pl.col(use_col).replace(dict_map))
    impute_to = impute_to.with_columns(
        pl.when(pl.col(key_col).is_null())
        .then(pl.col("tmp"))
        .otherwise(pl.col(key_col))
        .alias(key_col)
    ).drop("tmp")

    return impute_to


def get_n_unique_values(df: pl.DataFrame | pl.LazyFrame, use_col="subject_id"):
    unique_vals = df.select(use_col).unique()
    return (
        unique_vals.count().collect(streaming=True).item()
        if type(unique_vals) == pl.LazyFrame
        else unique_vals.count().item()
    )


def get_feature_list(subjects_root_path):
    features = (
        pl.concat(
            [
                pl.scan_csv(f).select(pl.col("label"))
                for f in glob.glob(os.path.join(subjects_root_path, "*", "events.csv"))
            ],
            how="diagonal",
        )
        .unique()
        .collect()
    )
    return sorted(features.get_column("label").to_list())


def get_pop_means(subjects_root_dir: str, output_dir: str = None) -> dict | None:
    """

    Uses events data to get population-level mean for features in events.csv

    Args:
        subjects_root_dir (str): Path to subject-level directory.
        output_dir (str, optional): Path to output directory to save csv. Defaults to None.

    Returns:
        dict | None: Save a csv file containing features and mean values to output_dir or return as a dict for mapping.
    """
    events_files = glob(os.path.join(subjects_root_dir, "*", "events.csv"))

    # Use polars to scan all csvs
    events = pl.concat(
        [
            pl.scan_csv(f, null_values=["___"]).select(["value", "label"])
            for f in events_files
        ]
    )

    mean_values = (
        events.group_by(pl.col("label"))
        .agg(pl.col("value").mean())
        .drop_nulls(subset="value")
        .collect(streaming=True)
    )

    # Write to disk
    if output_dir is not None:
        mean_values.write_csv(os.path.join(output_dir, "mean_values.csv"))

    # Or return mapping as a dictionary
    return dict(mean_values.iter_rows())


def scale_numeric_features(data, numeric_cols=None, over=None):
    # Normalise/scale specified cols using MinMax scaling
    if over is None:
        scaled = data.select(
            (pl.col(numeric_cols) - pl.col(numeric_cols).min())
            / (pl.col(numeric_cols).max() - pl.col(numeric_cols).min())
        )
    else:
        # compute min max of cols over another groupby col e.g., subject_id or label
        scaled = data.select(
            (pl.col(numeric_cols) - pl.col(numeric_cols).min())
            / (pl.col(numeric_cols).max() - pl.col(numeric_cols).min()).over(over)
        )

    # ensure all scaled features are floats to 1 d.p
    scaled = scaled.with_columns(pl.all().round(1))
    return data.select(pl.col("*").exclude(numeric_cols)).hstack(scaled)


def preview_data(path_to_pkl):
    with open(path_to_pkl, "rb") as f:
        data_dict = pickle.load(f)
    example_id = list(data_dict.keys())[-1]
    print(f"Example data:{data_dict[example_id]}")


def read_from_txt(filepath):
    with open(filepath) as f:
        data = [str(line.strip()) for line in f.readlines()]
    return data
