import os
from glob import glob

import polars as pl


def get_n_unique_values(df: pl.DataFrame | pl.LazyFrame, use_col="subject_id"):
    unique_vals = df.select(use_col).unique()
    return (
        len(unique_vals.collect())
        if type(unique_vals) == pl.LazyFrame
        else len(unique_vals)
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
