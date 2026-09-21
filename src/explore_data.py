import argparse
import os
import sys

import polars as pl
import toml

import utils.exploration as m4exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data exploration functionality for MIMIC-IV v3.1."
    )
    parser.add_argument(
        "ehr_path",
        type=str,
        help="Directory containing pre-extracted MIMIC-IV EHR data.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config toml file containing lookup fields for grouping.",
        default="../config/targets.toml",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Directory where summary tables and plots should be written.",
        required=True,
    )
    parser.add_argument(
        "--display_dict_path",
        "-d",
        type=str,
        help="Path to dictionary for display names of features.",
        default="../config/feat_name_map.json",
    )
    parser.add_argument(
        "--bhc_fname",
        "-b",
        type=str,
        help="File name for BHC distribution plot.",
        default="bhc_dist_by_outcome.png",
    )
    parser.add_argument(
        "--pval_adjust",
        "-p",
        type=str,
        help="Method for p-value adjustment in summary tables and distribution plots. Defaults to 'bonferroni'.",
        default="bonferroni",
    )
    parser.add_argument(
        "--pval_test",
        type=str,
        help="Test type for comparing distribution of BHC token lengths. Defaults to 't-test welch'.",
        default="t-test_welch",
    )
    parser.add_argument(
        "--max_i",
        "-i",
        type=int,
        help="Number of rows argument for plotting function. Defaults to 2.",
        default=2,
    )
    parser.add_argument(
        "--max_j",
        "-j",
        type=int,
        help="Number of columns argument for plotting function. Defaults to 2.",
        default=2,
    )
    parser.add_argument(
        "--rot",
        "-r",
        type=int,
        help="Rotation degrees for ticks in plotting function. Defaults to 0.",
        default=0,
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

    args, _ = parser.parse_known_args()
    if not os.path.exists(args.ehr_path):
        print(f"Path to EHR data {args.ehr_path} does not exist.")
        sys.exit(1)

    ed_pts = pl.read_csv(os.path.join(args.ehr_path, "ehr_static.csv"))
    if args.lazy:
        ed_pts = ed_pts.lazy()

    if not os.path.exists(args.output_path):
        print(f"Creating output directory for data exploration at {args.output_path}")
        os.makedirs(args.output_path)

    config = toml.load(args.config)
    outcomes = config["outcomes"]["labels"]
    outcome_labels = config["outcomes"]["display"]
    attributes = config["attributes"]["labels"]
    attribute_labels = config["attributes"]["display"]
    nn_attr = config["attributes"]["nonnormal"]
    categorical = config["attributes"]["categorical"]
    age_bins = config["age"]["bins"]
    age_labels = config["age"]["labels"]

    if len(categorical) == 0:
        categorical = None

    print(
        "Generating summary tables and plots across all defined outcomes and attributes."
    )
    ed_pts = m4exp.assign_age_groups(
        ed_pts, bins=age_bins, labels=age_labels, use_lazy=args.lazy
    )
    for outcome, label in zip(outcomes, outcome_labels, strict=False):
        print(f"Generating summary table one by {outcome}..")
        m4exp.get_table_one(
            ed_pts,
            outcome,
            label,
            args.output_path,
            args.display_dict_path,
            sensitive_attr_list=attribute_labels,
            nn_attr=nn_attr,
            verbose=args.verbose,
            adjust_method=args.pval_adjust,
            cat_cols=categorical,
        )
    print("Plotting outcome distribution by sensitive attributes..")
    for attribute, label in zip(attributes, attribute_labels, strict=False):
        m4exp.plot_outcome_dist_by_sensitive_attr(
            ed_pts,
            attribute,
            label,
            args.output_path,
            outcomes,
            outcome_labels,
            maxi=args.max_i,
            maxj=args.max_j,
            rot=args.rot,
            figsize=(8, 8),
        )
    print("Plotting age distribution by sensitive attributes..")
    for attribute, label in zip(attributes, attribute_labels, strict=False):
        m4exp.plot_age_dist_by_sensitive_attr(
            ed_pts,
            attribute,
            label,
            args.output_path,
            outcomes,
            outcome_labels,
            labels=age_labels,
            maxi=args.max_i,
            maxj=args.max_j,
            rot=args.rot,
            figsize=(8, 8),
        )
    print("Plotting distribution of BHC token lengths by outcome and attributes..")
    m4exp.plot_token_length_by_attribute(
        ed_pts,
        args.output_path,
        attributes,
        attribute_labels,
        maxi=args.max_i,
        maxj=args.max_j,
        rot=args.rot,
        figsize=(8, 13),
        test_type=args.pval_test,
    )
    m4exp.plot_token_length_by_attribute(
        ed_pts,
        args.output_path,
        outcomes,
        outcome_labels,
        out_fname=args.bhc_fname,
        maxi=args.max_i,
        maxj=args.max_j,
        rot=args.rot,
        figsize=(8, 8),
        test_type=args.pval_test,
        gr_pairs={
            "in_hosp_death": [("N", "Y")],
            "ext_stay_7": [("N", "Y")],
            "non_home_discharge": [("N", "Y")],
            "icu_admission": [("N", "Y")],
        },
        outcome_mode=True,
    )
    print("Finished data exploration.")
