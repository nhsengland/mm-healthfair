import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import shap
import toml
from datasets import MIMIC4Dataset
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    false_positive_rate,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from utils.functions import read_from_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data.",
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

    config = toml.load(args.config)
    los_threshold = config["model"]["threshold"]

    # Get training and validation ids
    test_ids = read_from_txt(args.test) if args.test is not None else None

    print("Loading dataset...")
    test_set = MIMIC4Dataset(
        args.data_path,
        "test",
        ids=test_ids,
        los_thresh=los_threshold,
        static_only=True,
    )
    test_set.print_label_dist()

    x_test = []
    y_test = []
    for data in test_set:
        x, y = data
        x_test.append(x[0])
        y_test.append(y[0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    print("Evaluating on validation data...")
    y_hat = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_hat)
    bacc = balanced_accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    print("Predicting on validation...")
    print("Performance summary:", [acc, bacc, auc, auprc])

    ### Explain
    if args.explain:
        # Visualise important features
        features = test_set.get_feature_list()
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(20, 10))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()

        # Create shap plot
        explainer = shap.TreeExplainer(model)

        # Plot single waterfall plot

        # Correct classification (TP)
        tps = np.argwhere(np.logical_and(y_hat == 1, y_test == 1))

        if len(tps) > 0:
            tp = tps[0][0]

            plt.figure(figsize=(12, 4))
            plt.title(
                f"Truth: {int(y_test[tp])}, Predict: {int(model.predict(x_test[tp].reshape(1,-1)))}, Prob: {round(model.predict_proba(x_test[tp].reshape(1,-1))[:,1][0], 2)}"
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
                f"Truth: {int(y_test[fn])}, Predict: {int(model.predict(x_test[fn].reshape(1,-1)))}, Prob: {round(model.predict_proba(x_test[fn].reshape(1,-1))[:,1][0], 2)}"
            )
            shap.bar_plot(
                explainer(x_test[fn])[:, 1].values,
                feature_names=features,
                max_display=20,
            )
            plt.show()

        # Plot summary over all test
        start = time.time()
        shap_values = explainer(x_test)
        print(time.time() - start)

        plt.figure()
        shap.summary_plot(shap_values[:, :, 1], feature_names=features, max_display=20)
        plt.show()

    if args.fairness:
        #### Fairness evaluation using fairlearn API

        # Get feature for all test_ids
        protected_features = ["gender_0", "race_0", "insurance_0"]
        feature_values = test_set.view_features(protected_features)

        metrics = {
            "accuracy": accuracy_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
        }

        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_hat,
            sensitive_features=feature_values,
        )
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 1],
            colormap="Accent",
            legend=False,
            figsize=[12, 8],
            title=f"Fairness evaluation of {[x.upper() for x in protected_features]}",
        )
        plt.show()
