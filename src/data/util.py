import pandas as pd
import polars as pl


def dataframe_from_csv(path, header=0, chunksize=None, usecols=None):
    return pd.read_csv(path, header=header, chunksize=chunksize, usecols=usecols)


def melt_table(table, id_vars: list = None, value_vars: list = None):
    return pd.melt(table, id_vars=id_vars, value_vars=value_vars)


def get_n_unique_values(df: pl.DataFrame, use_col="subject_id"):
    unique_vals = df.select(use_col).unique()
    return (
        len(unique_vals.collect())
        if type(unique_vals) == pl.LazyFrame
        else len(unique_vals)
    )
