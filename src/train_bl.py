import argparse

import numpy as np
import toml
from datasets import MIMIC4Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from utils.functions import read_from_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.toml",
        help="Path to config toml file containing parameters.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Whether to use cpu. Defaults to gpu"
    )
    parser.add_argument("--ids", nargs="?", default=None, help="List of ids to use")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb for logging. Defaults to False",
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    device = "gpu" if not args.cpu else "cpu"
    use_wandb = args.wandb
    los_threshold = config["model"]["threshold"]

    if args.ids is not None:
        # Create training and validation splits based on hadm_ids
        hadm_ids = read_from_txt(args.ids)
        train_ids, val_ids = train_test_split(hadm_ids, test_size=0.2)
    else:
        train_ids = None
        val_ids = None

    print("Creating dataset...")

    training_set = MIMIC4Dataset(
        args.data_path,
        "train",
        ids=train_ids,
        los_thresh=los_threshold,
        static_only=True,
    )

    x_train = []
    y_train = []
    for data in training_set:
        x, y = data
        x_train.append(x[0])
        y_train.append(y[0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    validation_set = MIMIC4Dataset(
        args.data_path,
        "val",
        ids=val_ids,
        los_thresh=los_threshold,
        static_only=True,
    )

    x_val = []
    y_val = []
    for data in validation_set:
        x, y = data
        x_val.append(x[0])
        y_val.append(y[0])

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    params = [
        {
            "n_estimators": [10, 100, 5000],
            "criterion": ["gini", "entropy", "log_loss"],
            "class_weight": [None, "balanced"],
        }
    ]

    model = GridSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_grid=params,
        scoring=["balanced_accuracy", "roc_auc", "accuracy"],
        refit="balanced_accuracy",
        cv=3,
    )
    print("Training via Grid Search..")
    model.fit(x_train, y_train)

    print("Best params:", model.best_params_)
    print("Best score:", model.best_score_)

    y_hat = model.predict(x_val)
    prob = model.predict_proba(x_val)[:, 1]

    acc = accuracy_score(y_val, y_hat)
    bacc = balanced_accuracy_score(y_val, y_hat)
    auc = roc_auc_score(y_val, prob)
    print("Predicting on validation...")
    print("Performance summary:", [acc, bacc, auc])
