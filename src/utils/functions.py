import os
from glob import glob

import pandas as pd
import polars as pl


def dataframe_from_csv(
    path: str, header: int = 0, usecols: list | tuple = None
) -> pd.DataFrame:
    """

    Read dataframe from csv file with kwargs.

    Args:
        path (str): Path to csv.
        header (int, optional): Header row. Defaults to 0.
        usecols (list | tuple, optional): Columns to use. Defaults to None (reads all).

    Returns:
        pd.DataFrame: Returns pandas dataframe.
    """
    return pd.read_csv(path, header=header, usecols=usecols)


def melt_table(table, id_vars: list = None, value_vars: list = None) -> pd.DataFrame:
    return pd.melt(table, id_vars=id_vars, value_vars=value_vars)


def count_rows(table_path) -> int:
    total_rows = 0
    for chunk in dataframe_from_csv(table_path, chunksize=1000, usecols=[0]):
        total_rows += chunk.index.size
    return total_rows


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

    events = (
        events.group_by(pl.col("label"))
        .agg(pl.col("value").mean())
        .drop_nulls(subset="value")
        .collect(streaming=True)
    )

    # Write to disk
    if output_dir is not None:
        events.write_csv(os.path.join(output_dir, "mean_values.csv"))

    # Or return mapping as a dictionary
    return events.to_dict()
