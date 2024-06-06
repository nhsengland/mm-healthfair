import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap
import toml
from datasets import MIMIC4Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
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
    args = parser.parse_args()

    config = toml.load(args.config)
    los_threshold = config["model"]["threshold"]

    # Get training and validation ids
    test_ids = read_from_txt(args.test) if args.test is not None else None

    print("Creating dataset...")
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
    print("Predicting on validation...")
    print("Performance summary:", [acc, bacc, auc])

    # Visualise important features
    plt.figure(figsize=(20, 10))
    features = test_set.get_feature_list()
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()

    # Create shap plot
    explainer = shap.Explainer(model, x_test)
    shap_values = explainer(x_test)

    # Plot first value
    plt.figure()
    shap.waterfall_plot(shap_values[0], max_display=20)
    plt.show()
