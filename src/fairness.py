import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import toml
from fairlearn.metrics import (
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)

import utils.exploration as m4exp
from utils.fairness_utils import (
    get_bootstrapped_fairness_measures,
    get_fairness_summary,
    plot_bar_metric_frame,
    plot_fairness_by_age,
)
from utils.functions import load_pickle, save_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal fairness evaluator pipeline."
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Evaluation path for obtaining risk dictionary as .pkl file (generated from evaluate.py)."
        "Must include the directory name as (outcome_name)_(fusion_type)_(modalities), containing the .pkl file named as pf_(outcome_name)_(fusion_type)_(modalities).pkl.",
        default="../outputs/evaluation",
    )
    parser.add_argument(
        "--fair_path",
        "-f",
        type=str,
        help="Directory to store explanation plots.",
        default="../outputs/fairness",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing the saved model metadata.",
        default="logs/nhs-mm-healthfair",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Directory pointing to the saved model checkpoint (must be inside model_dir).",
        default="ext_stay_7_None_timeseries",
    )
    parser.add_argument(
        "--attr_path",
        "-a",
        type=str,
        help="Directory containing attributes metadata (original ehr_static.csv).",
        default="../outputs/ext_data/ehr_static.csv",
    )
    parser.add_argument(
        "--outcome",
        "-o",
        type=str,
        help="Binary outcome to use for multimodal evaluation (one of the labels in targets.toml)."
        "Defaults to prediction of extended hospital stay.",
        default="ext_stay_7",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="../config/targets.toml",
        help="Path to config toml file containing lookup fields and outcomes.",
    )
    parser.add_argument(
        "--group_by_age",
        action="store_true",
        help="If true, will generate a trajectory lineplot of the fairness measures across each age group (as assigned in targets.toml).",
    )
    parser.add_argument(
        "--plot_errors",
        action="store_true",
        help="If true, will use the Fairlearn API to generate a grouped barplot of error measures over each attribute (as specified in targets.toml).",
    )
    parser.add_argument(
        "--across_models",
        action="store_true",
        help="If true, will generate a trajectory lineplot of the fairness measures across multiple models (specified in paths within targets.toml).",
    )
    parser.add_argument(
        "--threshold_method",
        "-th",
        type=str,
        default="yd",
        help="Method to use for thresholding positive and negative classes under class imbalance. "
        "Options are 'yd' (Youden's J statistic) or 'f1' (Maximum achievable F1-score).",
    )
    parser.add_argument(
        "--boot_samples",
        "-b",
        type=int,
        default=1000,
        help="Number of bootstrap samples to use for calculating confidence intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for bootstrapping. Defaults to 42."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity.",
    )

    args = parser.parse_args()
    targets = toml.load(args.targets)
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    age_bins = targets["age"]["bins"]
    age_labels = targets["age"]["labels"]
    threshold_method = args.threshold_method
    ### For Pre-loading risk dictionaries if using --across_models
    model_paths = targets["paths"]["model_paths"]
    model_names = targets["paths"]["model_names"]
    model_colors = targets["paths"]["model_colors"]
    ### Model-specific setup
    model_path = os.path.join(
        os.path.join(args.model_dir, args.model_path), args.model_path + ".ckpt"
    )
    risk_path = os.path.join(
        os.path.join(args.eval_path, args.model_path), "pf_" + args.model_path + ".pkl"
    )
    fair_path = os.path.join(args.fair_path, args.model_path)
    if not os.path.exists(fair_path) and not args.across_models:
        os.makedirs(fair_path)
    ### Infer fusion method and modalities from model_path
    modalities = []
    for arg in args.model_path.split("_"):
        if "static" in arg:
            modalities.append("static")
        elif "timeseries" in arg:
            modalities.append("timeseries")
        elif "notes" in arg:
            modalities.append("notes")
    fusion_method = "None"
    if "mag" in args.model_path:
        fusion_method = "IF-mag"
    elif "concat" in args.model_path:
        fusion_method = "IF-concat"
    if args.outcome not in outcomes and not args.across_models:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    outcome_idx = outcomes.index(args.outcome)
    ### Set plotting style
    plt.rcParams.update(
        {"font.size": 16, "font.weight": "normal", "font.family": "serif"}
    )
    print("------------------------------------------")
    print("MMHealthFair: Multimodal Fairness analysis")
    print("------------------------------------------")
    if not args.across_models:
        print(f'Evaluating fairness for outcome "{outcomes_disp[outcome_idx]}"')
        print(f"Modalities used: {modalities}")
        print(f"Fusion method: {fusion_method}")
    print(f"Fairness across multiple models: {args.across_models}")
    ### Get test ids
    if (len(model_path) == 0) and (not args.across_models):
        print(f"No model found at {args.model_path}. Exiting..")
        sys.exit()

    if not os.path.exists(risk_path) and not args.across_models:
        print(f"No risk dictionary found at {risk_path}. Exiting..")
        sys.exit()

    ### Get risk predictions
    risk_dict = load_pickle(risk_path)
    test_ids = risk_dict["test_ids"]
    y_test = risk_dict["y_test"]
    y_prob = risk_dict["y_prob"]
    ### Use Youden's J statistic of F1-max rather than sigmoid for best discrimination threshold as classes are imbalanced
    if threshold_method == "yd":
        thres = risk_dict["yd_idx"]
    elif threshold_method == "f1":
        thres = risk_dict["f1_thres"]
    y_hat = np.where(y_prob > thres, 1, 0)
    risk_dict["y_hat"] = y_hat
    ### Get sensitive attributes data
    print("Reading attributes metadata for fairness analysis...")
    if not os.path.exists(args.attr_path):
        print("Attributes metadata not found. Exiting...")
        sys.exit()
    ehr_static = pl.read_csv(args.attr_path).filter(
        pl.col("subject_id").is_in(list(map(int, test_ids)))
    )
    ### If --across_models, load results dictionary and generate summary
    if args.across_models:
        print("Generating Fairness summary across all selected models...")
        res_all = {}
        for model, path in zip(model_names, model_paths, strict=False):
            r_path = os.path.join(
                os.path.join(args.fair_path, path), "pf_" + path + ".pkl"
            )
            res_dict = load_pickle(r_path)
            fair_keys = [key for key in res_dict.keys() if key.startswith("fair_")]
            fair_dict = {key: res_dict[key] for key in fair_keys}
            res_all[model] = fair_dict
        # print(res_all)
        ## Plot fairness measures across all models
        get_fairness_summary(
            res_all,
            model_names,
            model_colors,
            attribute_labels=attr_disp,
            outcome=outcomes_disp[outcome_idx],
            output_path=f"{args.fair_path}/fair_full_across_models.png",
        )
        print("Evaluation complete.")
        sys.exit()

    ### For estimation of grouped fairness by age
    if args.group_by_age:
        age_cols = {
            k: risk_dict[k] for k in ("test_ids", "risk_quantile", "y_test", "y_hat")
        }
        age_df = pl.DataFrame(age_cols)
        ehr_age = ehr_static.select(["subject_id", "anchor_age"])
        age_df = age_df.join(ehr_age, left_on="test_ids", right_on="subject_id")
        age_df = m4exp.assign_age_groups(
            age_df, bins=age_bins, labels=age_labels, use_lazy=False
        )
        aq_dict = {}

    # Get feature for all test_ids from metadata
    for pf, disp in zip(attributes, attr_disp, strict=False):
        attr_pf = ehr_static.select(pl.col(pf))
        if pf == "insurance":
            # Replace "Other" with "Medicare" in the insurance column
            attr_pf = attr_pf.with_columns(
                pl.when(pl.col(pf) == "Other")
                .then(pl.lit("Medicare").cast(pl.Utf8))
                .otherwise(pl.col(pf))
                .alias(pf)
            )
        if args.verbose:
            print("---------------------------------------")
            print(f"Evaluating fairness by {disp}...")
        if args.plot_errors:
            group_metrics = {
                "FPR (1-Sensitivity)": false_positive_rate,
                "FNR (1-Specificity)": false_negative_rate,
                "Selection Rate": selection_rate,
                "Subject Count": count,
            }
            plot_bar_metric_frame(
                group_metrics,
                y_test,
                y_hat,
                attr_pf,
                attribute=disp,
                seed=args.seed,
                save_path=os.path.join(fair_path, f"error_by_{pf}.png"),
            )

        # Global fairness measures
        dpr, eor, eop = get_bootstrapped_fairness_measures(
            y_test,
            y_hat,
            attr_pf,
            n_boot=args.boot_samples,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(
            f"Demographic Parity: {dpr[0]:.3f} (95% CI: [{dpr[1]:.3f}, {dpr[2]:.3f}])"
        )
        print(f"Equalized Odds: {eor[0]:.3f} (95% CI: [{eor[1]:.3f}, {eor[2]:.3f}])")
        print(f"Equal Opportunity: {eop[0]:.3f} (95% CI: [{eop[1]:.3f}, {eop[2]:.3f}])")
        # Append measures to risk dictionary
        risk_dict["fair_" + disp] = {
            "DPR": round(dpr[0], 3),
            "DPR_CI": [round(dpr[1], 3), round(dpr[2], 3)],
            "EQO": round(eor[0], 3),
            "EQO_CI": [round(eor[1], 3), round(eor[2], 3)],
            "EOP": round(eop[0], 3),
            "EOP_CI": [round(eop[1], 3), round(eop[2], 3)],
        }

        if args.group_by_age:
            if args.verbose:
                print("Getting fairness measures by age...")
            for aq in age_labels:
                if args.verbose:
                    print(f"{disp} by age group {aq}...")
                # Get values for current risk group
                age_df_rq = age_df.filter(pl.col("age_group") == aq)
                y_test_rq = age_df_rq["y_test"].to_numpy()
                y_hat_rq = age_df_rq["y_hat"].to_numpy()
                attr_pf_rq = ehr_static.filter(
                    pl.col("subject_id").is_in(list(map(int, age_df_rq["test_ids"])))
                ).select(pl.col(pf))
                # Local fairness measures
                dpr, eor, eop = get_bootstrapped_fairness_measures(
                    y_test_rq,
                    y_hat_rq,
                    attr_pf_rq,
                    n_boot=args.boot_samples,
                    seed=args.seed,
                    skip_ci=True,
                    verbose=args.verbose,
                )
                aq_dict[pf + "_aq_" + aq] = {
                    "DPR": round(dpr[0], 3),
                    "EQO": round(eor[0], 3),
                    "EOP": round(eop[0], 3),
                    "Size": len(y_test_rq),
                }
                # print(aq_dict)

    if args.group_by_age:
        print("Plotting fairness measures by age...")
        plot_fairness_by_age(
            aq_dict,
            age_labels,
            os.path.join(fair_path, "dpr_by_age.png"),
            attributes,
            attr_disp,
            measure="DPR",
            measure_label="Demographic Parity",
        )
        plot_fairness_by_age(
            aq_dict,
            age_labels,
            os.path.join(fair_path, "eor_by_age.png"),
            attributes,
            attr_disp,
            measure="EQO",
            measure_label="Equalized Odds",
        )
        plot_fairness_by_age(
            aq_dict,
            age_labels,
            os.path.join(fair_path, "eop_by_age.png"),
            attributes,
            attr_disp,
            measure="EOP",
            measure_label="Equal Opportunity",
        )

    save_pickle(risk_dict, fair_path, f"pf_{args.model_path}.pkl")
    print(f"Saved results dictionary to {fair_path}/pf_{args.model_path}.pkl")
    print("Evaluation complete.")
