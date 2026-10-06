import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    equal_opportunity_ratio,
    equalized_odds_ratio,
)
from scipy.stats import norm
from tqdm import tqdm


def plot_bar_metric_frame(
    metrics: dict,
    y_test: np.ndarray,
    y_hat: np.ndarray,
    attr_df: pl.DataFrame,
    attribute: str,
    save_path: str,
    figsize=None,
    nrows=2,
    ncols=2,
    seed=0,
):
    """
    Plot an error bar chart for the given metric frame using Fairlearn.

    Args:
        metrics (dict): Dictionary of metric functions to compute.
        y_test (np.ndarray): Ground truth labels.
        y_hat (np.ndarray): Predicted labels.
        attr_df (pl.DataFrame): Sensitive attribute DataFrame.
        attribute (str): Name of the sensitive attribute.
        save_path (str): Path to save the plot.
        figsize (tuple, optional): Figure size.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        seed (int): Random seed.

    Returns:
        None
    """
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_hat,
        sensitive_features=attr_df,
        random_state=seed,
    )
    figsize = figsize if figsize else (12, 8)

    plt.figure(figsize=figsize)
    fig = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[nrows, ncols],
        colormap="Pastel2",
        legend=False,
        figsize=(figsize),
        title=f"Error estimates by {attribute}",
    )

    axes = fig.flatten()
    for i in range(len(axes)):
        plt.sca(axes[i])
        plt.xticks(rotation=45, ha="center")
        plt.xlabel("")
        if i < len(axes) - 1:
            axes[i].set_ylim(0.0, 1.0)
            axes[i].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
            )

    fig = plt.gcf()
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Error plot saved to {save_path}.")


# Extract bias-corrected confidence intervals using BCa method (example of effects and variations: https://www.erikdrysdale.com/bca_python/)
def bias_corrected_ci(bootstrap_samples, observed_value):
    """
    Calculate bias-corrected and accelerated (BCa) confidence intervals for bootstrap samples.

    Args:
        bootstrap_samples (array-like): Bootstrap sample values.
        observed_value (float): Observed value for bias correction.

    Returns:
        tuple: (lower_bound, upper_bound) confidence interval.
    """
    sorted_samples = np.sort(bootstrap_samples)
    # Calculate z0 (bias correction factor)
    z0 = norm.ppf((np.sum(sorted_samples < observed_value) + 0.5) / len(sorted_samples))
    # Calculate a (acceleration factor) using jackknife
    jackknife_estimates = [
        np.mean(np.delete(sorted_samples, i)) for i in range(len(sorted_samples))
    ]
    mean_jackknife = np.mean(jackknife_estimates)
    a = np.sum((mean_jackknife - jackknife_estimates) ** 3) / (
        6 * (np.sum((mean_jackknife - jackknife_estimates) ** 2) ** 1.5)
    )
    # Adjust percentiles
    alpha = [0.025, 0.975]  # For a 95% CI
    adjusted_percentiles = norm.cdf(
        z0 + (z0 + norm.ppf(alpha)) / (1 - a * (z0 + norm.ppf(alpha)))
    )
    # If any value is null, set to alpha
    if np.isnan(adjusted_percentiles[0]):
        adjusted_percentiles = alpha
    lower_bound = np.percentile(sorted_samples, adjusted_percentiles[0] * 100)
    upper_bound = np.percentile(sorted_samples, adjusted_percentiles[1] * 100)
    return lower_bound, upper_bound


def get_bootstrapped_fairness_measures(
    y_test: np.ndarray,
    y_hat: np.ndarray,
    attr_pf: pl.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    skip_ci: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Compute bootstrapped fairness measures (Demographic Parity, Equalized Odds, Equal Opportunity)
    and their confidence intervals.

    Args:
        y_test (np.ndarray): Ground truth labels.
        y_hat (np.ndarray): Predicted labels.
        attr_pf (pl.DataFrame): Sensitive attribute DataFrame.
        n_boot (int): Number of bootstrap samples.
        seed (int): Random seed.
        skip_ci (bool): If True, skip CI calculation.
        verbose (bool): If True, print additional information.

    Returns:
        tuple: (dpr_full, eor_full, eop_full) where each contains (mean, lower_CI, upper_CI).
    """
    global_metrics = {
        "Demographic Parity": demographic_parity_ratio,
        "Equalized Odds": equalized_odds_ratio,
        "Equal Opportunity": equal_opportunity_ratio,
    }
    # Initialize random state
    rng = np.random.default_rng(seed)
    if verbose:
        print(attr_pf.to_pandas().value_counts())
    # Store bootstrap results
    bootstrap_results = {metric: [] for metric in global_metrics}
    # Perform bootstrapping (random sampling with stratification)
    for _ in tqdm(range(n_boot)):
        # Sample with stratification
        stratified_indices = []
        for group in np.unique(attr_pf.to_numpy()):
            group_indices = np.where(attr_pf.to_numpy() == group)[0]
            stratified_indices.extend(
                rng.choice(group_indices, size=len(group_indices), replace=True)
            )
        stratified_indices = np.array(stratified_indices)
        y_test_sample = y_test[stratified_indices]
        y_hat_sample = y_hat[stratified_indices]
        attr_pf_sample = attr_pf[stratified_indices, :]

        # Calculate metrics for the sample
        dp_boot = demographic_parity_ratio(
            y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample
        )
        eor_boot = equalized_odds_ratio(
            y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample
        )
        eop_boot = equal_opportunity_ratio(
            y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample
        )
        # Replace 0 values with 0.001 to avoid division by zero in ratios
        eor_boot = max(eor_boot, 0.001)
        eop_boot = max(eop_boot, 0.001)

        # Store results
        bootstrap_results["Demographic Parity"].append(dp_boot)
        bootstrap_results["Equalized Odds"].append(eor_boot)
        bootstrap_results["Equal Opportunity"].append(eop_boot)

    # Calculate overall metrics from bootstrap results (under imbalance assumption)
    dpr = np.mean(bootstrap_results["Demographic Parity"])
    eor = np.mean(bootstrap_results["Equalized Odds"])
    eop = np.mean(bootstrap_results["Equal Opportunity"])
    dp_ci = eor_ci = eop_ci = (0, 0)

    # Skip CI calculation if using risk quantiles
    if not skip_ci:
        dp_ci = bias_corrected_ci(bootstrap_results["Demographic Parity"], dpr)
        eor_ci = bias_corrected_ci(bootstrap_results["Equalized Odds"], eor)
        eop_ci = bias_corrected_ci(bootstrap_results["Equal Opportunity"], eop)

    dpr_full = (dpr, dp_ci[0], dp_ci[1])
    eor_full = (eor, eor_ci[0], eor_ci[1])
    eop_full = (eop, eop_ci[0], eop_ci[1])

    return dpr_full, eor_full, eop_full


def plot_fairness_by_age(
    aq_dict: dict,
    age_labels: list,
    out_path: str,
    attributes: list,
    attribute_labels: list,
    figsize: tuple = (11, 8),
    measure: str = "DPR",
    measure_label: str = "Demographic Parity",
):
    """
    Plot fairness measures by age group for multiple sensitive attributes.

    Args:
        aq_dict (dict): Dictionary containing fairness metrics by age group.
        age_labels (list): List of age group labels.
        out_path (str): Path to save the plot.
        attributes (list): List of sensitive attribute names.
        attribute_labels (list): List of attribute display names.
        figsize (tuple): Figure size.
        measure (str): Key for the fairness measure to plot.
        measure_label (str): Display label for the fairness measure.

    Returns:
        None
    """
    # Define attributes and their labels
    attributes = ["gender", "race_group", "insurance", "marital_status"]
    attribute_labels = ["Sex", "Ethnicity", "Insurance", "Marital Status"]

    # Define unique colors for each age group
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_labels)))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    # Set suptitle
    fig.suptitle(f"{measure_label} by age and sensitive group.", fontsize=22)
    fig.supylabel(measure_label, fontsize=20)
    axes = axes.flatten()

    # Plot DPR for each attribute
    for i, (attr, label) in enumerate(zip(attributes, attribute_labels, strict=False)):
        # Filter keys for the current attribute
        attr_keys = {key: value for key, value in aq_dict.items() if attr in key}
        m_values = [attr_keys[f"{attr}_aq_{age}"][measure] for age in age_labels]
        m_counts = [aq_dict[f"{attr}_aq_{age}"]["Size"] for age in age_labels]

        # Plot on the corresponding subplot
        for j, age in enumerate(age_labels):
            axes[i].plot(
                age,
                m_values[j],
                marker="o",
                color=colors[j],
                label=f"Age {age}, N={m_counts[j]}",
            )

        axes[i].set_title(f"{label}", fontsize=20)
        axes[i].set_ylim(-0.05, 1)
        axes[i].grid(True)
        axes[i].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
        )
        if i == 1:
            axes[i].legend(
                title="Age Group", loc="upper left", bbox_to_anchor=(1, 1), fontsize=14
            )
        else:
            axes[i].legend().set_visible(False)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Fairness plot saved to {out_path}")
    plt.close()


def get_fairness_summary(
    res_all: dict,
    models: list,
    colors: list,
    attribute_labels: list,
    figsize: tuple = (13, 8),
    nrows: int = 2,
    ncols: int = 2,
    outcome: str = "Extended Stay",
    output_path: str = "fair_full_across_models.png",
):
    """
    Plot grouped barplots for fairness metrics (DPR, EQO, EOP) with 95% CI for each model.

    Args:
        res_all (dict): Dictionary containing fairness metrics for each model.
        models (list): List of model names.
        colors (list): List of colors for each model.
        attribute_labels (list): List of attribute display names.
        figsize (tuple): Figure size.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        outcome (str): Name of the outcome.
        output_path (str): Path to save the plot.

    Returns:
        None
    """
    # Define attributes and their labels
    attributes = ["fair_" + item for item in attribute_labels]

    # Define metrics and their labels
    metrics = ["DPR", "EQO", "EOP"]

    # Set default colors if not provided
    if not colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    # Iterate over attributes and plot each subplot
    for i, (attr, label) in enumerate(zip(attributes, attribute_labels, strict=False)):
        ax = axes[i]
        x = np.arange(len(metrics))  # Positions for metrics
        width = 0.15  # Width of each bar

        # Plot bars for each model
        for j, model in enumerate(models):
            metric_values = [res_all[model][attr][metric] for metric in metrics]
            ci_values = [res_all[model][attr][f"{metric}_CI"] for metric in metrics]
            lower_bounds = [
                metric - ci[0]
                for metric, ci in zip(metric_values, ci_values, strict=False)
            ]
            upper_bounds = [
                ci[1] - metric
                for metric, ci in zip(metric_values, ci_values, strict=False)
            ]

            # Plot bars with error bars
            ax.bar(
                x + j * width,
                metric_values,
                width,
                label=model,
                color=colors[j],
                yerr=[lower_bounds, upper_bounds],
                capsize=5,
                alpha=0.8,
            )

        # Set subplot title and labels
        ax.set_title(label, fontsize=17)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=17)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))

        # Add legend to the second subplot
        if i == 1:
            ax.legend(
                title="Model", loc="upper left", bbox_to_anchor=(1, 1), fontsize=16
            )

    # Adjust layout and save the plot
    fig.suptitle(f"Fairness Estimates For {outcome} Prediction", fontsize=19)
    fig.supxlabel("Fairness Type", fontsize=19)
    fig.supylabel("Estimate", fontsize=19)
    plt.tight_layout()  # Adjust layout to fit legend
    plt.savefig(output_path)
    print(f"Global fairness summary saved to {output_path}")
    plt.close()
