import pandas as pd


def dataframe_from_csv(path, header=0, chunksize=None, usecols=None):
    return pd.read_csv(path, header=header, chunksize=chunksize, usecols=usecols)


def melt_table(table, id_vars: list = None, value_vars: list = None):
    return pd.melt(table, id_vars=id_vars, value_vars=value_vars)


def count_rows(table_path):
    total_rows = 0
    for chunk in dataframe_from_csv(table_path, chunksize=1000, use_cols=[0]):
        total_rows += chunk.index.size
    return total_rows
