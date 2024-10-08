import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import toml
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from lightning.pytorch import Trainer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import concat
from torch.utils.data import DataLoader

from datasets import CollateTimeSeries, MIMIC4Dataset
from models import MMModel
from utils.functions import load_pickle, read_from_txt
from utils.preprocessing import transform_race

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the directory containing processed data.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.toml",
        help="Path to config toml file containing parameters.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to stays.csv containing hadm_id and raw data. Specify if processed data is in different directory to stays.csv.",
    )
    parser.add_argument(
        "--test", default=None, help="List of ids to use for evaluation."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Whether to generate explainability plots.",
    )
    parser.add_argument(
        "--fairness", action="store_true", help="Whether to evaluate fairness."
    )
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, "processed_data.pkl")
    metadata_path = (
        os.path.join(args.data_dir, "stays.csv")
        if args.metadata is None
        else args.metadata
    )

    config = toml.load(args.config)
    los_threshold = config["model"]["threshold"]

    # if loading from .ckpt (deep learning) set static_only to False else assume static only model (RF)
    model_type = "fusion" if os.path.splitext(args.model_path)[1] == ".ckpt" else "rf"
    static_only = True if model_type == "rf" else False

    # Get training and validation ids
    test_ids = read_from_txt(args.test) if args.test is not None else None

    test_set = MIMIC4Dataset(
        data_path,
        "test",
        ids=test_ids,
        los_thresh=los_threshold,
        static_only=static_only,
    )
    test_set.print_label_dist()

    if model_type == "rf":
        print("Loading dataset...")
        x_test = []
        y_test = []
        for data in test_set:
            x, y = data
            x_test.append(x[0])
            y_test.append(y[0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        model = load_pickle(args.model_path)
        print("Evaluating on test data...")
        y_hat = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]

    elif model_type == "fusion":
        test_dataloader = DataLoader(
            test_set,
            batch_size=8,
            collate_fn=CollateTimeSeries(),
        )

        model = MMModel.load_from_checkpoint(checkpoint_path=args.model_path)
        print("Evaluating on test data...")

        trainer = Trainer(accelerator="gpu")
        output = trainer.predict(model, dataloaders=test_dataloader)
        default_thresh = 0.5
        prob = concat([out[0] for out in output])
        y_hat = np.where(prob > default_thresh, 1, 0)
        y_test = concat([out[1] for out in output])

    acc = accuracy_score(y_test, y_hat)
    bacc = balanced_accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    print("Predicting on validation...")
    print("Performance summary:", [acc, bacc, auc, auprc])

    ### Explain
    if args.explain:
        if model_type == "rf":
            # Visualise important features
            features = test_set.get_feature_list()
            importances = model.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

            # Create shap plot
            explainer = shap.TreeExplainer(model, x_test)

            # Plot single waterfall plot
            # Correct classification (TP)
            tps = np.argwhere(np.logical_and(y_hat == 1, y_test == 1))

            if len(tps) > 0:
                tp = tps[0][0]

                plt.figure(figsize=(12, 4))
                plt.title(
                    f"Truth: {int(y_test[tp])}, Predict: {int(y_hat[tp])}, Prob: {round(prob[tp], 2)}"
                )
                shap.bar_plot(
                    explainer(x_test[tp])[:, 1].values,
                    feature_names=features,
                    max_display=20,
                )
                plt.show()

            # Incorrect (FN)
            fns = np.argwhere(np.logical_and(y_hat == 0, y_test == 1))

            if len(fns) > 0:
                fn = fns[0][0]

                plt.figure(figsize=(12, 4))
                plt.title(
                    f"Truth: {int(y_test[fn])}, Predict: {int(y_hat[fn])}, Prob: {round(prob[fn], 2)}"
                )
                shap.bar_plot(
                    explainer(x_test[fn])[:, 1].values,
                    feature_names=features,
                    max_display=20,
                )
                plt.show()

            # Plot summary over all test subjects
            start = time.time()
            shap_values = explainer(x_test, check_additivity=False)
            print(time.time() - start)

            plt.figure()
            shap.summary_plot(
                shap_values[:, :, 1], feature_names=features, max_display=20
            )
            plt.show()

        elif model_type == "fusion":
            # get first collated batch (fixed size and num of samples)
            batch = next(iter(test_dataloader))

            for i in range(2):
                features = test_set.get_feature_list(f"dynamic_{i}")

                x_test = batch[2][i]
                explainer = shap.DeepExplainer(model.embed_timeseries[i], x_test)

                # Plot summary over all test subjects for single timepoint (t=0)
                shap_values = explainer.shap_values(x_test, check_additivity=False)

                plt.figure()
                shap.summary_plot(
                    shap_values.mean(axis=3)[:, 0, :],
                    feature_names=features,
                    features=x_test[:, 0, :],
                )
                plt.show()

    if args.fairness:
        #### Fairness evaluation using fairlearn API

        # Get feature for all test_ids from metadata
        protected_features = ["gender", "race", "insurance", "marital_status"]

        for pf in protected_features:
            metadata = (
                pl.scan_csv(metadata_path)
                .filter(pl.col("hadm_id").is_in(list(map(int, test_ids))))
                .select(pf)
                .collect()
            )

            # group races
            if pf == "race":
                metadata = transform_race(metadata)
                race_groups = {
                    0: "Unknown/Other",
                    1: "Asian",
                    2: "Black",
                    3: "Hispanic",
                    4: "White",
                }
                metadata = metadata.with_columns(pl.col("race").replace(race_groups))

            if pf == "marital_status":
                metadata = metadata.with_columns(
                    pl.col("marital_status").replace({None: "Unspecified"})
                )

            metrics = {
                "accuracy": accuracy_score,
                "false positive rate": false_positive_rate,
                "false negative rate": false_negative_rate,
                "selection rate": selection_rate,
                "count": count,
            }

            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=y_test,
                y_pred=y_hat,
                sensitive_features=metadata,
            )

            metric_frame.by_group.plot.bar(
                subplots=True,
                layout=[3, 2],
                colormap="Pastel2",
                legend=False,
                figsize=[12, 8],
                title="Fairness evaluation",
                xlabel=pf,
            )

            # fairness
            eor = equalized_odds_ratio(y_test, y_hat, sensitive_features=metadata)
            eod = equalized_odds_difference(y_test, y_hat, sensitive_features=metadata)
            dpr = demographic_parity_ratio(y_test, y_hat, sensitive_features=metadata)

            print(f"EOR: {eor},  EOD:{eod}, DPR: {dpr}")

        plt.show()
