import pandas as pd


def dataframe_from_csv(path, header=0):
    return pd.read_csv(path, header=header)


def melt_table(table, id_vars: list = None, value_vars: list = None):
    return pd.melt(table, id_vars=id_vars, value_vars=value_vars)
