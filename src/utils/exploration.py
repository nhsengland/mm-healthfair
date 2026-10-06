import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator
from tableone import TableOne


def get_table_one(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    outcome: str,
    outcome_label: str,
    output_path: str = "../outputs/reference",
    disp_dict_path: str = "../outputs/reference/feat_name_map.json",
    sensitive_attr_list: list = "None",
    nn_attr: list = "None",
    adjust_method="bonferroni",
    cat_cols: list = None,
    verbose: bool = False,
) -> TableOne:
    """
    Generate a baseline patient summary table (Table 1) with adjusted p-values grouped by outcome.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        outcome (str): Outcome variable name.
        outcome_label (str): Display name for the outcome.
        output_path (str): Directory to save the HTML summary.
        disp_dict_path (str): Path to JSON mapping feature names to display names.
        sensitive_attr_list (list): List of sensitive attribute names.
        nn_attr (list): List of non-normal columns.
        adjust_method (str): Method for p-value adjustment.
        cat_cols (list): List of categorical columns.
        verbose (bool): If True, print summary information.

    Returns:
        TableOne: Generated TableOne summary object.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect().to_pandas()
    else:
        ed_pts = ed_pts.to_pandas()
    ### Load dictionary containing display names for features
    with open(disp_dict_path) as f:
        disp_dict = json.load(f)
    ### Infer categorical data if not specified
    if cat_cols is None:
        cat_cols = [el for el in disp_dict.values() if el not in nn_attr]
    ed_disp = ed_pts.rename(columns=disp_dict)
    ed_disp = ed_disp[disp_dict.values()]
    ### Code categorical columns
    for col in cat_cols:
        if col not in sensitive_attr_list:
            ed_disp[col] = np.where(ed_disp[col] == 1, "Yes", "No")
    ### Generate Table 1 summary with p-values for target outcome
    if verbose:
        print(
            f"Generating table summary by {outcome_label} with prevalence {(ed_disp[outcome_label].value_counts(normalize=True).iloc[1] * 100).round(2)}%"
        )
    tb_one_hd = TableOne(
        ed_disp,
        categorical=[col for col in cat_cols if outcome_label != col],
        nonnormal=nn_attr,
        groupby=outcome_label,
        overall=True,
        pval=True,
        htest_name=True,
        tukey_test=True,
        decimals=0,
        pval_adjust=adjust_method,
    )
    tb_one_hd.to_html(os.path.join(output_path, f"table_one_{outcome}.html"))
    print(
        f"Saved table summary grouped by {outcome_label} to table_one_{outcome}.html."
    )
    return tb_one_hd


def assign_age_groups(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    age_col: str = "anchor_age",
    bins: list = None,
    labels: list = None,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Assign age groups to patients based on age column and specified bins/labels.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        age_col (str): Name of the age column.
        bins (list): List of bin edges for age groups.
        labels (list): List of labels for age groups.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame: DataFrame with an added 'age_group' column.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ed_pts = ed_pts.with_columns(
        pl.when(pl.col(age_col) < bins[1])
        .then(pl.lit(labels[0]))
        .when((pl.col(age_col) >= bins[1]) & (pl.col(age_col) < bins[2]))
        .then(pl.lit(labels[1]))
        .when((pl.col(age_col) >= bins[2]) & (pl.col(age_col) < bins[3]))
        .then(pl.lit(labels[2]))
        .when((pl.col(age_col) >= bins[3]) & (pl.col(age_col) < bins[4]))
        .then(pl.lit(labels[3]))
        .otherwise(pl.lit(labels[4]))
        .alias("age_group")
    )
    return ed_pts.lazy() if use_lazy else ed_pts


def get_age_table_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_name: str,
    outcome: str,
    value_name_col: str,
) -> pl.DataFrame:
    """
    Transform the dataset into long format to group samples with the outcome by age group and sensitive attribute.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_name (str): Sensitive attribute column name.
        outcome (str): Outcome variable name.
        value_name_col (str): Name for the value column in the melted DataFrame.

    Returns:
        pd.DataFrame: Long-format DataFrame with counts and percentages by group.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ed_pts = ed_pts.to_pandas()

    ed_inh_long = pd.melt(
        ed_pts[ed_pts[outcome] == "Y"],
        id_vars=[attr_name],
        value_vars=["age_group"],
        value_name=value_name_col,
    )
    ed_inh_long = (
        ed_inh_long.groupby([attr_name, value_name_col], observed=False)
        .size()
        .reset_index(name="# Patients")
    )
    ed_y_cts = (
        ed_inh_long.groupby(attr_name)["# Patients"]
        .apply(lambda x: x.sum())
        .reset_index()
        .rename(columns={"# Patients": "Total"})
    )
    ed_inh_long = ed_inh_long.merge(ed_y_cts, how="left", on=attr_name)
    ed_inh_long["Percentage"] = round(
        ed_inh_long["# Patients"] / ed_inh_long["Total"], 4
    )

    return ed_inh_long


def plot_outcome_dist_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_col: str,
    attr_xlabel: str,
    output_path: str = "../outputs/reference",
    outcome_list: list = None,
    outcome_title: list = None,
    outcome_legend: dict = None,
    maxi: int = 2,
    maxj: int = 2,
    rot: int = 0,
    figsize: tuple = (8, 6),
    palette: list = None,
):
    """
    Plot the distribution of health outcomes by a specified sensitive attribute.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_col (str): Sensitive attribute column name.
        attr_xlabel (str): Label for the sensitive attribute display label.
        output_path (str): Directory to save the plot.
        outcome_list (list): List of outcome variable names.
        outcome_title (list): List of outcome display names.
        outcome_legend (dict): Mapping of outcome titles to legend labels.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        rot (int): Rotation angle for x-axis labels.
        figsize (tuple): Figure size.
        palette (list): List of colors for plotting.

    Returns:
        None
    """
    ### Edit when customising outcomes
    outcome_legend = {
        "In-hospital Death": ["No", "Yes"],
        "Extended Hospital Stay": ["No", "Yes"],
        "Non-home Discharge": ["No", "Yes"],
        "ICU Admission": ["No", "Yes"],
    }
    palette = ["#1f77b4", "#ff7f0e"]
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ### Set plot config
    plt.rcParams.update(
        {"font.size": 12, "font.weight": "normal", "font.family": "serif"}
    )
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(f"Health outcome distribution by {attr_xlabel}", fontsize=16)
    fig.supxlabel(attr_xlabel, fontsize=14)
    fig.supylabel("Percentage of patients with ED attendance", fontsize=14)
    outcome_idx = 0
    if attr_col == "race_group":
        ed_pts = ed_pts.with_columns(
            pl.when(pl.col(attr_col) == "Hispanic/Latino")
            .then(pl.lit("His/Latino"))
            .otherwise(pl.col(attr_col))
            .alias(attr_col)
        )
    unique_values = ed_pts.select(attr_col).unique().to_series().to_list()[::-1]
    for i in range(maxi):
        for j in range(maxj):
            ed_gr = (
                ed_pts.groupby([attr_col, outcome_list[outcome_idx]])
                .agg(pl.count().alias("count"))
                .to_pandas()
            )
            total_counts = ed_gr.groupby(attr_col)["count"].transform("sum")
            ed_gr["percentage"] = ed_gr["count"] / total_counts * 100
            # Plot relative percentages
            sns.barplot(
                data=ed_gr,
                x=attr_col,
                y="percentage",
                hue=outcome_list[outcome_idx],
                ax=ax[i][j],
                palette=palette,
                order=unique_values,
            )
            handles = [
                Patch(
                    color=palette[k],
                    label=outcome_legend[outcome_title[outcome_idx]][k],
                )
                for k in range(len(outcome_legend[outcome_title[outcome_idx]]))
            ]
            ax[i][j].legend(handles=handles, title=outcome_title[outcome_idx])
            ax[i][j].set_xlabel("")
            ax[i][j].set_ylabel("")
            ax[i][j].yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f"{x:.0f}%")
            )
            if j == 1:
                ax[i][j].tick_params(left=False)
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha="center")
            outcome_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"outcome_dist_by_{attr_col}.png"))
    print(f"Saved plot to outcome_dist_by_{attr_col}.png.")


def plot_age_dist_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_col: str,
    attr_xlabel: str,
    output_path: str = "../outputs/reference",
    outcome_list: list = None,
    outcome_title: list = None,
    maxi: int = 2,
    maxj: int = 2,
    colors: list = None,
    labels: list = None,
    figsize: tuple = (12, 8),
    rot: int = 0,
):
    """
    Plot the distribution of age groups by a specified sensitive attribute for patients with adverse events.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_col (str): Sensitive attribute column name.
        attr_xlabel (str): Label for the sensitive attribute display name.
        output_path (str): Directory to save the plot.
        outcome_list (list): List of outcome variable names.
        outcome_title (list): List of outcome display names.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        colors (list): List of colors for age groups.
        labels (list): List of age group labels.
        figsize (tuple): Figure size.
        rot (int): Rotation angle for x-axis labels.

    Returns:
        None
    """
    ### Edit when customising outcomes
    colors = ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"]
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ### Set plot config
    plt.rcParams.update(
        {"font.size": 12, "font.weight": "normal", "font.family": "serif"}
    )
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        "Age distribution by sensitive variable in patients with adverse event.",
        fontsize=16,
    )
    fig.supxlabel(attr_xlabel, fontsize=14)
    fig.supylabel("Percentage of patients with ED attendance", fontsize=14)

    if attr_col == "race_group":
        ed_pts = ed_pts.with_columns(
            pl.when(pl.col(attr_col) == "Hispanic/Latino")
            .then(pl.lit("His/Latino"))
            .otherwise(pl.col(attr_col))
            .alias(attr_col)
        )
    ### Recode outcome variables as 'N', 'Y' for plotting
    for outcome in outcome_list:
        ed_pts = ed_pts.with_columns(
            pl.when(pl.col(outcome) == 1)
            .then(pl.lit("Y"))
            .otherwise(pl.lit("N"))
            .alias(outcome)
        )
    outcome_idx = 0
    lg_flag = False
    for i in range(maxi):
        for j in range(maxj):
            lg_flag = True if outcome_idx == 1 else False
            ed_long = get_age_table_by_sensitive_attr(
                ed_pts, attr_col, outcome_list[outcome_idx], "age_" + attr_xlabel
            )

            # Reorder rows in 'age_'+attr_xlabel according to order in list 'labels'
            ed_long["age_" + attr_xlabel] = pd.Categorical(
                ed_long["age_" + attr_xlabel], categories=labels, ordered=True
            )
            ed_long = ed_long.sort_values(by=[attr_col, "age_" + attr_xlabel])
            ax[i][j] = pd.pivot_table(
                ed_long[[attr_col, "age_" + attr_xlabel, "Percentage"]],
                columns="age_" + attr_xlabel,
                index=attr_col,
                sort=True,
                observed=False,
            ).plot(
                title=outcome_title[outcome_idx],
                kind="bar",
                stacked=True,
                figsize=figsize,
                ax=ax[i][j],
                legend=lg_flag,
                color=colors,
            )
            # print(ed_long_pivot.head())
            # ed_long_pivot.plot(kind='bar', stacked=True, ax=ax[i][j], legend=lg_flag, color=colors)
            ax[i][j].set_title(outcome_title[outcome_idx])
            if lg_flag:
                ax[i][j].legend(title="Age Group", labels=labels, bbox_to_anchor=(1, 1))
            if j == 1:
                ax[i][j].tick_params(left=False)
            ax[i][j].set_xlabel("")
            ax[i][j].set_ylabel("")
            ax[0][0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha="center")
            outcome_idx += 1

    plt.tight_layout()

    plt.savefig(os.path.join(output_path, f"age_dist_by_{attr_col}.png"))
    print(f"Saved plot to age_dist_by_{attr_col}.png.")


def plot_token_length_by_attribute(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    output_path: str = "../outputs/reference",
    sensitive_attr_list: list = None,
    attr_title: list = None,
    out_fname: str = "bhc_dist_by_attr.png",
    maxi: int = 2,
    maxj: int = 2,
    figsize: tuple = (8, 6),
    rot: int = 0,
    ylim: tuple = (0, 12),
    gr_pairs: dict = None,
    suptitle: str = "BHC token length by sensitive variable in patients with ED attendance.",
    outcome_mode: bool = False,
    unique_value_order: list = None,
    adjust_method: str = "bonferroni",
    test_type: str = "t-test_welch",
):
    """
    Display violin plots of aggregated BHC length by sensitive attribute, with statistical annotation.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        output_path (str): Directory to save the plot.
        sensitive_attr_list (list): List of sensitive attribute names.
        attr_title (list): List of attribute display names.
        out_fname (str): Output filename for the plot.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        figsize (tuple): Figure size.
        rot (int): Rotation angle for x-axis labels.
        ylim (tuple): Y-axis limits.
        gr_pairs (dict): Dictionary of group pairs for statistical annotation.
        suptitle (str): Figure title.
        outcome_mode (bool): If True, relabel categories for outcome mode.
        unique_value_order (list): Order of unique values for plotting.
        adjust_method (str): Method for p-value adjustment.
        test_type (str): Statistical test type.

    Returns:
        None
    """
    ### Edit when customising outcomes
    unique_value_order = ["N", "Y"]
    if gr_pairs is None:
        gr_pairs = {
            "gender": [("M", "F")],
            "insurance": [
                ("Medicare", "Medicaid"),
                ("Medicare", "Private"),
                ("Medicaid", "Private"),
            ],
            "race_group": [
                ("White", "Black"),
                ("White", "Hispanic/Latino"),
                ("White", "Asian"),
                ("Black", "Hispanic/Latino"),
                ("Black", "Asian"),
                ("Hispanic/Latino", "Asian"),
            ],
            "marital_status": [
                ("Divorced", "Single"),
                ("Divorced", "Married"),
                ("Divorced", "Widowed"),
                ("Single", "Married"),
                ("Single", "Widowed"),
                ("Married", "Widowed"),
            ],
        }
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()

    ### log transform target tokens as they are right-skewed
    ed_pts = ed_pts.with_columns(
        (pl.col("num_target_tokens").log1p()).alias("num_target_tokens_lg")
    )
    ## Exclude other categories
    ed_pts = ed_pts.filter(pl.col("race_group") != "Other")
    ed_pts = ed_pts.filter(pl.col("insurance") != "Other")

    ### Set plot config
    plt.rcParams.update(
        {"font.size": 12, "font.weight": "normal", "font.family": "serif"}
    )
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(suptitle, fontsize=16)
    fig.supylabel("Log-transformed token length", fontsize=14)
    attr_idx = 0
    for i in range(maxi):
        for j in range(maxj):
            ### If exploring health outcomes, relabel the categories
            if outcome_mode:
                ed_pts = ed_pts.with_columns(
                    pl.when(pl.col(sensitive_attr_list[attr_idx]) == 1)
                    .then(pl.lit("Y"))
                    .otherwise(pl.lit("N"))
                    .alias(sensitive_attr_list[attr_idx])
                )
                ax[i][j] = sns.violinplot(
                    data=ed_pts.to_pandas(),
                    x=sensitive_attr_list[attr_idx],
                    y="num_target_tokens_lg",
                    ax=ax[i][j],
                    order=unique_value_order,
                )
            else:
                ax[i][j] = sns.violinplot(
                    data=ed_pts.to_pandas(),
                    x=sensitive_attr_list[attr_idx],
                    y="num_target_tokens_lg",
                    ax=ax[i][j],
                )
            ax[i][j].set_xlabel(attr_title[attr_idx])
            ax[i][j].set_ylabel("")
            if j == 1:
                ax[i][j].tick_params(left=False)
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha="center")
            plt.ylim(ylim)
            ### Annotate significant differences
            print(
                f"Annotating significant differences for {sensitive_attr_list[attr_idx]}"
            )
            annot = Annotator(
                ax[i][j],
                data=ed_pts.to_pandas(),
                x=sensitive_attr_list[attr_idx],
                y="num_target_tokens_lg",
                pairs=gr_pairs[sensitive_attr_list[attr_idx]],
            )
            annot.configure(
                test=test_type,
                text_format="star",
                loc="outside",
                verbose=0,
                comparisons_correction=adjust_method,
            )
            annot._pvalue_format.pvalue_thresholds = [
                [0.001, "***"],
                [0.01, "**"],
                [0.1, "*"],
                [1, "ns"],
            ]
            annot.apply_and_annotate()
            attr_idx += 1

    plt.tight_layout(pad=0.5)
    tgt = os.path.join(output_path, out_fname)
    plt.savefig(tgt)
    print(f"Saved plot to {tgt}.")
