import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import toml
from lightning.pytorch import Trainer
from torch import concat
from torch.utils.data import DataLoader

from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset
from models import MMModel
from utils.eval_utils import (
    get_all_roc_pr_summary,
    get_pr_performance,
    get_roc_performance,
    plot_calibration_curve,
    plot_learning_curve,
    plot_pr,
    plot_roc,
    rank_prediction_quantiles,
)
from utils.functions import load_pickle, save_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal performance evaluation pipeline."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_feat.pkl",
    )
    parser.add_argument(
        "--col_path",
        "-p",
        type=str,
        help="Path to the pickled column dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_cols.pkl",
    )
    parser.add_argument(
        "--ids_path",
        "-i",
        type=str,
        help="Directory containing test ids.",
        default="../outputs/prep_data",
    )
    parser.add_argument(
        "--attr_path",
        "-a",
        type=str,
        help="Directory containing attributes metadata (original ehr_static.csv).",
        default="../outputs/ext_data/ehr_static.csv",
    )
    parser.add_argument(
        "--eval_path",
        "-e",
        type=str,
        help="Directory to store performance plots.",
        default="../outputs/evaluation",
    )
    parser.add_argument(
        "--outcome",
        "-o",
        type=str,
        help="Binary outcome to use for multimodal learning (one of the labels in targets.toml)."
        "Defaults to prediction of in-hospital death.",
        default="in_hosp_death",
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
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="../config/model.toml",
        help="Path to config toml file containing model training parameters.",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="../config/targets.toml",
        help="Path to config toml file containing lookup fields and outcomes.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for quantile analysis and calibration curve.",
    )
    parser.add_argument(
        "--strat_by_attr",
        action="store_true",
        help="Show stratified quantile analysis by attribute.",
    )
    parser.add_argument(
        "--group_models",
        action="store_true",
        help="Generate Precision-Recall summary across all models specified in targets.toml.",
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
    config = toml.load(args.config)
    targets = toml.load(args.targets)

    ### Config setup for single-model evaluation
    if not args.group_models:
        model_path = os.path.join(
            os.path.join(args.model_dir, args.model_path), args.model_path + ".ckpt"
        )
        loss_path = os.path.join(
            os.path.join(args.model_dir, args.model_path), "losses.csv"
        )
        eval_path = os.path.join(args.eval_path, args.model_path)
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        num_workers = config["data"]["num_workers"]
        modalities = []
        for arg in args.model_path.split("_"):
            if "static" in arg:
                modalities.append("static")
            elif "timeseries" in arg:
                modalities.append("timeseries")
            elif "notes" in arg:
                modalities.append("notes")

        static_only = (
            True if (len(modalities) == 1) and ("static" in modalities) else False
        )
        with_notes = True if "notes" in modalities else False
        fusion_method = "None"
        if "mag" in args.model_path:
            fusion_method = "IF-mag"
        elif "concat" in args.model_path:
            fusion_method = "IF-concat"

    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    batch_size = config["data"]["batch_size"]
    ### For Pre-loading models if using --group_models
    model_paths = targets["paths"]["model_paths"]
    model_names = targets["paths"]["model_names"]
    model_colors = targets["paths"]["model_colors"]
    if args.outcome not in outcomes:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    outcome_idx = outcomes.index(args.outcome)
    ### Set plotting style
    plt.rcParams.update(
        {"font.size": 13, "font.weight": "normal", "font.family": "serif"}
    )
    print("------------------------------------------")
    print("MMHealthFair: Multimodal evaluation pipeline")
    ### If --group_models, load results dictionary and generate summary
    if args.group_models:
        print("Generating Precision-Recall summary across all selected models...")
        res_all = []
        for _, path in zip(model_names, model_paths, strict=False):
            r_path = os.path.join(
                os.path.join(args.eval_path, path), "pf_" + path + ".pkl"
            )
            res_dict = load_pickle(r_path)
            res_all.append(res_dict)

        get_all_roc_pr_summary(
            res_all,
            model_names,
            model_colors,
            output_roc_path=f"{args.eval_path}/roc_full_across_models.png",
            output_pr_path=f"{args.eval_path}/pr_full_across_models.png",
        )
        print("Evaluation complete.")
        sys.exit()
    print(f'Evaluating performance for outcome "{outcomes_disp[outcome_idx]}"')
    print(f"Modalities used: {modalities}")
    print(f"Fusion method: {fusion_method}")
    print("------------------------------------------")

    # if loading from .ckpt (deep learning) set static_only to False else assume static only model (RF)
    # model_type = "fusion" if os.path.splitext(args.model_path)[1] == ".ckpt" else "rf"
    # static_only = True if model_type == "rf" else False

    # Get test ids
    if (
        len(
            glob.glob(
                os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv")
            )
        )
        == 0
    ):
        print(f"No test ids found for outcome {args.outcome}. Exiting..")
        sys.exit()

    if (len(glob.glob(model_path)) == 0) and (not args.group_models):
        print(f"No model found at {model_path}. Exiting..")
        sys.exit()

    test_ids = (
        pl.read_csv(os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv"))
        .select("subject_id")
        .to_numpy()
        .flatten()
    )

    if args.strat_by_attr:
        print("Reading attributes metadata for stratified quantile analysis...")
        if not os.path.exists(args.attr_path):
            print("Attributes metadata not found. Exiting...")
            sys.exit()

    ehr_static = pl.read_csv(args.attr_path).to_pandas()

    test_set = MIMIC4Dataset(
        args.data_path,
        args.col_path,
        "test",
        ids=test_ids,
        static_only=static_only,
        with_notes=with_notes,
    )
    test_set.print_label_dist()
    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True if num_workers > 0 else False,
    )

    model = MMModel.load_from_checkpoint(checkpoint_path=model_path)
    print("Evaluating on test data...")
    trainer = Trainer(accelerator="gpu")
    output = trainer.predict(model, dataloaders=test_dataloader)
    default_thresh = 0.5
    prob = np.array(concat([out[0] for out in output])).flatten().astype(np.float32)
    y_hat = np.where(prob > default_thresh, 1, 0)
    y_test = np.array(concat([out[1] for out in output])).flatten().astype(np.int8)

    ### Performance summary
    print("------------------------------------------")
    print("Plotting learning curve...")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    if (len(glob.glob(loss_path)) != 0) and (not args.group_models):
        plot_learning_curve(
            loss_path, output_path=f"{eval_path}/lc_{args.model_path}.png"
        )
    else:
        print(
            f"No losses.csv found at {loss_path}. Skipping learning curve plot for {args.model_path}."
        )
    print("------------------------------------------")
    print("ROC performance report:")
    print("------------------------------------------")
    bin_labels, res_roc_dict = get_roc_performance(y_test, prob, verbose=args.verbose)
    print("------------------------------------------")
    print("Precision-Recall performance report:")
    print("------------------------------------------")
    res_pr_dict = get_pr_performance(
        y_test, prob, bin_labels, opt_f1=True, verbose=args.verbose
    )
    print("------------------------------------------")
    print("Getting ROC/PR curves...")
    plot_roc(
        y_test,
        prob,
        outcome=outcomes_disp[outcome_idx],
        result_dict=res_roc_dict,
        output_path=f"{eval_path}/roc_{args.model_path}.png",
    )
    plot_pr(
        y_test,
        prob,
        outcome=outcomes_disp[outcome_idx],
        result_dict=res_pr_dict,
        output_path=f"{eval_path}/pr_{args.model_path}.png",
    )
    print("Plotting calibration curve...")
    plot_calibration_curve(
        y_test,
        prob,
        outcome=outcomes_disp[outcome_idx],
        n_bins=20,
        output_path=f"{eval_path}/calib_{args.model_path}.png",
    )
    print("------------------------------------------")
    print("Prediction quantile analysis:")
    print("------------------------------------------")
    res_pd_dict = rank_prediction_quantiles(
        y_test,
        prob,
        n_bins=args.n_bins,
        outcome=outcomes_disp[outcome_idx],
        output_path=f"{eval_path}/rstrat_{args.model_path}.png",
        by_attribute=args.strat_by_attr,
        attrs=attributes,
        attr_disp=attr_disp,
        test_ids=test_ids,
        attr_features=ehr_static,
        verbose=args.verbose,
    )
    print("------------------------------------------")
    print("Saving results dictionary...")
    res_dict = res_roc_dict | res_pr_dict | res_pd_dict
    ### Record ground-truths and probabilities
    res_dict["y_test"] = y_test
    res_dict["y_prob"] = prob
    res_dict["test_ids"] = test_ids
    save_pickle(res_dict, eval_path, f"pf_{args.model_path}.pkl")
    print(f"Saved results dictionary to {eval_path}/pf_{args.model_path}.pkl")
    print("Evaluation complete.")
