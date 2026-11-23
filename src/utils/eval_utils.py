import glob

import confidenceinterval as cfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def plot_learning_curve(losses_path: str = None, output_path="learning_curve.png"):
    """
    Plot the learning curve (training and validation loss) from a CSV file.

    Args:
        losses_path (str): Path to CSV file containing the training and validation loss.
        output_path (str): Path to save the learning curve plot.

    Returns:
        None
    """
    if not glob.glob(losses_path):
        print(f"No file found at {losses_path}")
        return
    losses = pd.read_csv(losses_path)
    x_axis = losses["Epoch"]
    plt.figure(figsize=(6, 6))
    plt.plot(
        x_axis,
        losses["Train Loss"],
        color="darkorange",
        label="Training Loss",
        marker="o",
    )
    plt.plot(
        x_axis,
        losses["Validation Loss"],
        color="navy",
        label="Validation Loss",
        marker="o",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MMFair Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Learning curve saved to {output_path}")


def plot_roc(
    y_test: np.array,
    prob: np.array,
    output_path: str = "roc.png",
    result_dict: dict = None,
    outcome: str = "In-hospital Death",
):
    """
    Plot the ROC curve with AUC and 95% CI for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the ROC curve plot.
        result_dict (dict): Target performance dictionary for storing performance metrics.
        outcome (str): Name of the outcome being evaluated.

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_test, prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=1.5,
        label=f"AUC = {roc_auc:0.3f} [{result_dict['roc_lower']:.3f}, {result_dict['roc_upper']:.3f}]",
    )
    ### Get 95% CI for TPR by computing covariance matrix from Z-score
    tpr_se = np.sqrt((tpr * (1 - tpr)) / len(y_test))
    z = stats.norm.ppf(1 - 0.05 / 2)
    result_dict["tpr_lower"] = np.maximum(tpr - z * tpr_se, 0)
    result_dict["tpr_upper"] = np.minimum(tpr + z * tpr_se, 1)
    plt.fill_between(
        fpr,
        result_dict["tpr_lower"],
        result_dict["tpr_upper"],
        color="darkorange",
        alpha=0.3,
        label="95% CI",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"MMFair ROC Curve ({outcome})")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    print(f"ROC curve saved to {output_path}")


def plot_pr(
    y_test: np.array,
    prob: np.array,
    output_path: str = "pr_curve.png",
    result_dict: dict = None,
    outcome: str = "In-hospital Death",
):
    """
    Plot the Precision-Recall curve with AUC and 95% CI for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the PR curve plot.
        result_dict (dict): Target performance dictionary for storing performance metrics.
        outcome (str): Name of the outcome being evaluated.

    Returns:
        None
    """
    prevalence = np.sum(y_test) / len(y_test)
    precision, recall, _ = precision_recall_curve(y_test, prob, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 6))
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=1.5,
        label=f"AUC = {pr_auc:0.3f} [{result_dict['pr_lower']:.3f}, {result_dict['pr_upper']:.3f}]",
    )
    plt.plot([0, 1], [prevalence, prevalence], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR ({outcome}, prevalence {(100 * prevalence):.2f}%)")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    print(f"Precision-Recall curve saved to {output_path}")


def plot_calibration_curve(
    y_test: np.array,
    prob: np.array,
    output_path: str = "calib_curve.png",
    outcome: str = "In-hospital Death",
    n_bins: int = 10,
):
    """
    Plot the calibration curve for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels (0 or 1).
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the calibration curve plot.
        outcome (str): Name of the outcome being evaluated.
        n_bins (int): Number of bins to use for calibration.

    Returns:
        None
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test, prob, n_bins=n_bins, strategy="uniform"
    )

    # Plot calibration curve
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve ({outcome})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Calibration curve saved to {output_path}")


def expect_f1(y_prob: np.array, thres: int) -> float:
    """
    Calculate expected F1 score for a given threshold.

    Args:
        y_prob (np.array): Predicted probabilities.
        thres (float): Threshold for binary classification.

    Returns:
        float: Expected F1 score.
    """
    idx_tp = np.where(y_prob >= thres)[0]
    idx_fn = np.where(y_prob < thres)[0]
    tp = y_prob[idx_tp].sum()
    fn = y_prob[idx_fn].sum()
    fp = len(idx_tp) - tp
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0


def optimal_threshold(y_prob: np.array) -> float:
    """
    Calculate the optimal threshold for binary classification based on expected F1 score.

    Args:
        y_prob (np.array): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """
    y_prob = np.sort(y_prob[::-1])
    f1_scores = [expect_f1(y_prob, p) for p in y_prob]
    optimal_thres = y_prob[np.argmax(f1_scores)]
    return optimal_thres


def get_roc_performance(y_test: np.array, prob: np.array, verbose: bool = False):
    """
    Compute ROC performance summary based on Youden's J statistic for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        verbose (bool): If True, print detailed performance metrics.

    Returns:
        tuple: (bin_labels, res_dict_roc)
            bin_labels (np.array): Binary predictions using Youden's J threshold.
            res_dict_roc (dict): ROC statistics and confidence intervals.
    """
    fpr, tpr, th = roc_curve(y_test, prob, pos_label=1)
    res_dict_roc = {}
    ### Get Youden's J statistic
    J = np.argmax(tpr - fpr)
    yd = th[J]
    if verbose:
        print(f"Youden's J statistic: {yd:.3f}")
    bin_labels = np.asarray(prob >= yd, dtype=int)
    if verbose:
        print("Classification report for J threshold:")
        print(classification_report(y_test, bin_labels, target_names=["0", "1"]))

    aucss, ci = cfi.roc_auc_score(y_test, prob, confidence_level=0.95)
    ppv, cip = cfi.ppv_score(y_test, bin_labels, confidence_level=0.95)
    npv, cin = cfi.npv_score(y_test, bin_labels, confidence_level=0.95)
    tnr, cit = cfi.tnr_score(y_test, bin_labels, confidence_level=0.95)
    tpr, cis = cfi.tpr_score(y_test, bin_labels, confidence_level=0.95)
    if verbose:
        print(f"ROC-AUC with 95% CI: {aucss:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"PPV: {ppv:.3f} [{cip[0]:.3f}, {cip[1]:.3f}]")
        print(f"NPV: {npv:.3f} [{cin[0]:.3f}, {cin[1]:.3f}]")
        print(f"Specificity: {tnr:.3f} [{cit[0]:.3f}, {cit[1]:.3f}]")
        print(f"Sensitivity: {tpr:.3f} [{cis[0]:.3f}, {cis[1]:.3f}]")
    res_dict_roc["roc_auc"] = aucss
    res_dict_roc["roc_upper"] = ci[1]
    res_dict_roc["roc_lower"] = ci[0]
    res_dict_roc["yd_idx"] = yd
    return bin_labels, res_dict_roc


def get_pr_performance(
    y_test: np.array,
    prob: np.array,
    bin_labels: np.array,
    opt_f1: bool = True,
    verbose: bool = False,
):
    """
    Compute Precision-Recall performance metrics and confidence intervals.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        bin_labels (np.array): Binary predictions.
        opt_f1 (bool): If True, use optimal F1 threshold.
        verbose (bool): If True, print detailed performance metrics.

    Returns:
        dict: PR statistics and confidence intervals.
    """
    precision, recall, _ = precision_recall_curve(y_test, prob, pos_label=1)
    res_dict_pr = {}
    ### Get F1 score
    f1 = f1_score(y_test, bin_labels)
    thres = 0.5
    auc_score = auc(recall, precision)
    if opt_f1:
        thres = optimal_threshold(prob)
    if verbose:
        print(f"Optimal F1 score: {f1:.3f}")
        print(f"Threshold for optimal F1: {thres:.3f}")
        bin_labels = np.asarray(prob >= thres, dtype=int)
        f1 = f1_score(y_test, bin_labels)
        if verbose:
            print("Classification report for optimal F1 threshold:")
            print(classification_report(y_test, bin_labels, target_names=["0", "1"]))
        ppv, cip = cfi.ppv_score(y_test, bin_labels, confidence_level=0.95)
        npv, cin = cfi.npv_score(y_test, bin_labels, confidence_level=0.95)
        tnr, cit = cfi.tnr_score(y_test, bin_labels, confidence_level=0.95)
        tpr, cis = cfi.tpr_score(y_test, bin_labels, confidence_level=0.95)
        if verbose:
            print(f"PPV (Precision): {ppv:.3f} [{cip[0]:.3f}, {cip[1]:.3f}]")
            print(f"NPV (N-Precision): {npv:.3f} [{cin[0]:.3f}, {cin[1]:.3f}]")
            print(f"Specificity (TNR): {tnr:.3f} [{cit[0]:.3f}, {cit[1]:.3f}]")
            print(f"Sensitivity (TPR): {tpr:.3f} [{cis[0]:.3f}, {cis[1]:.3f}]")

    ### Get 95% CI for PR AUC by computing covariance matrix from Z-score
    fp = np.sum((bin_labels == 1) & (y_test == 0))
    tp = np.sum((bin_labels == 1) & (y_test == 1))
    fn = np.sum((bin_labels == 0) & (y_test == 1))
    prec_s = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_s = tp / (tp + fn) if (tp + fn) > 0 else 0
    se_prec = np.sqrt(prec_s * (1 - prec_s) / (tp + fp))
    se_rec = np.sqrt(recall_s * (1 - recall_s) / (tp + fn))
    z = stats.norm.ppf(1 - 0.05 / 2)
    cip = (prec_s - z * se_prec, prec_s + z * se_prec)
    cin = (recall_s - z * se_rec, recall_s + z * se_rec)
    pr_var = (cip[1] - cip[0]) ** 2 / 4
    rec_var = (cin[1] - cin[0]) ** 2 / 4
    cov_mat = [[pr_var, 0], [0, rec_var]]
    auc_se = np.sqrt(np.dot(np.dot([1, 1], cov_mat), [1, 1]))
    lb = auc_score - 1.96 * auc_se
    ub = auc_score + 1.96 * auc_se
    if verbose:
        print(f"PR-AUC with 95% CI: {auc_score:.3f} [{lb:.3f}, {ub:.3f}]")
    res_dict_pr["pr_auc"] = auc_score
    res_dict_pr["pr_upper"] = ub
    res_dict_pr["pr_lower"] = lb
    res_dict_pr["prec"] = prec_s
    res_dict_pr["recall"] = recall_s
    res_dict_pr["prevalence"] = np.sum(y_test) / len(y_test)
    res_dict_pr["f1_thres"] = thres
    return res_dict_pr


def get_all_roc_pr_summary(
    res_dicts: list,
    models: list,
    colors: list,
    output_roc_path: str = "roc_summary.png",
    output_pr_path: str = "pr_summary.png",
):
    """
    Plot summary ROC and PR curves for multiple models.

    Args:
        res_dicts (list): List of dictionaries with model results.
        models (list): List of model names.
        colors (list): List of colors for plotting.
        output_roc_path (str): Path to save ROC summary plot.
        output_pr_path (str): Path to save PR summary plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    for res_dict, model, color in zip(res_dicts, models, colors, strict=False):
        fpr, tpr, _ = roc_curve(res_dict["y_test"], res_dict["y_prob"], pos_label=1)
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=1.5,
            label=f"{model} (AUC = {res_dict['roc_auc']:.3f} [{res_dict['roc_lower']:.3f}, {res_dict['roc_upper']:.3f}])",
        )
        ### Get 95% CI for TPR by computing covariance matrix from Z-score
        tpr_se = np.sqrt((tpr * (1 - tpr)) / len(res_dict["y_test"]))
        z = stats.norm.ppf(1 - 0.05 / 2)
        tpr_lower = np.maximum(tpr - z * tpr_se, 0)
        tpr_upper = np.minimum(tpr + z * tpr_se, 1)
        plt.fill_between(fpr, tpr_lower, tpr_upper, color=color, alpha=0.3)
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (all models)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_roc_path)
    print(f"ROC summary saved to {output_roc_path}")

    plt.figure(figsize=(8, 5))
    for res_dict, model, color in zip(res_dicts, models, colors, strict=False):
        precision, recall, _ = precision_recall_curve(
            res_dict["y_test"], res_dict["y_prob"], pos_label=1
        )
        plt.plot(
            recall,
            precision,
            color=color,
            lw=1.5,
            label=f"{model} (AUC = {res_dict['pr_auc']:.3f} [{res_dict['pr_lower']:.3f}, {res_dict['pr_upper']:.3f}])",
        )

    plt.plot(
        [0, 1],
        [res_dict["prevalence"], res_dict["prevalence"]],
        color="black",
        lw=1.5,
        linestyle="--",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (all models)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_pr_path)
    print(f"PR summary saved to {output_pr_path}")


def rank_prediction_quantiles(
    y_test: np.array,
    prob: np.array,
    attrs: list,
    attr_disp: list,
    test_ids: list,
    n_bins: int = 10,
    outcome: str = "In-hospital Death",
    output_path: str = "risk_strat.png",
    by_attribute: bool = False,
    attr_features: pd.DataFrame = None,
    verbose: bool = False,
):
    """
    Rank predictions into quantiles and plot risk stratification, optionally stratified by attribute.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        attrs (list): List of attribute names for stratification.
        attr_disp (list): List of display names for attributes.
        test_ids (list): List of subject IDs for test set.
        n_bins (int): Number of quantiles.
        outcome (str): Name of the outcome being evaluated.
        output_path (str): Path to save the risk stratification plot.
        by_attribute (bool): If True, stratify by attribute.
        attr_features (pd.DataFrame): DataFrame with attribute features.
        verbose (bool): If True, print detailed information.

    Returns:
        dict: Appended patient risk quantiles.
    """
    if verbose:
        print(f"Ranking prediction quantiles for {outcome}..")
    res_dict = {}
    lkup_df = pd.DataFrame()
    lkup_df["prob"] = prob
    lkup_df["label"] = y_test
    lkup_df["quantile"] = pd.qcut(prob, n_bins, labels=False) + 1
    avg_resp = lkup_df["label"].mean() * 100
    dec_stats = lkup_df.groupby("quantile")["label"].sum().reset_index()
    samples = lkup_df.groupby("quantile")["label"].count().reset_index().iloc[:, 1:2]
    samples.columns = ["total"]
    dec_stats["rr"] = round((dec_stats["label"] / samples["total"]) * 100, 2)
    ### Plot risk stratification
    plt.figure(figsize=(6, 6))
    plt.bar(
        range(1, n_bins + 1),
        dec_stats["rr"],
        width=0.8,
        align="center",
        alpha=0.7,
        color="navy",
    )
    plt.axhline(
        y=avg_resp, color="crimson", linestyle="-", label=f"ARR: {avg_resp:.2f}%"
    )
    plt.xlabel("Risk quantile")
    plt.ylabel("Average response rate (% positive cases)")
    plt.title(f"Risk Stratification: {outcome}")
    plt.xticks(range(1, n_bins + 1))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.legend(loc="upper left")
    plt.savefig(output_path)
    print(f"Risk stratification plot saved to {output_path}")
    res_dict["10th_quantile_rr"] = dec_stats["rr"].iloc[-1]
    res_dict["risk_quantile"] = lkup_df["quantile"].values.tolist()
    #### If stratifying by attribute plot the risk quantile distribution for each attribute
    if by_attribute:
        lkup_df["subject_id"] = test_ids
        attr_features = attr_features[attr_features["subject_id"].isin(test_ids)]
        lkup_df = lkup_df.merge(attr_features, on="subject_id", how="left")
        for attr, disp in zip(attrs, attr_disp, strict=False):
            if verbose:
                print(f"Plotting stratified quantile plot for: {disp}")
            eval_long = pd.melt(
                lkup_df,
                id_vars=["quantile"],
                value_vars=[attr],
                value_name=f"{attr}_Value",
            )
            eval_long = (
                eval_long.groupby(["quantile", f"{attr}_Value"])["quantile"]
                .size()
                .reset_index(name="Count")
            )
            eval_y = (
                eval_long.groupby("quantile")["Count"]
                .apply(lambda x: x.sum())
                .reset_index()
                .rename(columns={"Count": "Total"})
            )
            eval_long = eval_long.merge(eval_y, on="quantile", how="left")
            eval_long["Percentage"] = round(eval_long["Count"] / eval_long["Total"], 5)
            ax = pd.pivot_table(
                eval_long[["quantile", f"{attr}_Value", "Percentage"]],
                index="quantile",
                columns=f"{attr}_Value",
            ).plot(
                kind="bar",
                stacked=True,
                figsize=(9, 5),
                title=f"Risk profiles: {outcome} by {disp}",
                colormap="tab10",
            )
            ax.legend(
                title=f"{disp}",
                labels=eval_long[f"{attr}_Value"].unique(),
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
            )
            plt.xticks(rotation=0, ha="center")
            plt.xlabel("Risk quantile")
            plt.ylabel(f"Distribution across {disp}")
            plt.tight_layout()

            tg_path = output_path.split(".png")
            plt.savefig(f"{tg_path[0]}_by_{attr}.png")
            print(f"Risk stratification plot saved to {tg_path[0]}_by_{attr}.png")

    return res_dict
