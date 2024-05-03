import polars as pl


def get_n_unique_values(df: pl.DataFrame, use_col="subject_id"):
    unique_vals = df.select(use_col).unique()
    return (
        len(unique_vals.collect())
        if type(unique_vals) == pl.LazyFrame
        else len(unique_vals)
    )
