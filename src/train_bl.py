import argparse

import numpy as np
import toml
from datasets import MIMIC4Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
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
        "--hp",
        action="store_true",
        help="Whether to use Grid Search CV for hyperparameter tuning.",
    )
    parser.add_argument(
        "--train", default=None, help="List of ids to use for training."
    )
    parser.add_argument(
        "--val", default=None, help="List of ids to use for validation."
    )

    args = parser.parse_args()

    config = toml.load(args.config)
    los_threshold = config["model"]["threshold"]
    with_hp = args.hp

    # Get training and validation ids
    train_ids = read_from_txt(args.train) if args.train is not None else None
    val_ids = read_from_txt(args.val) if args.val is not None else None

    print("Creating dataset...")

    training_set = MIMIC4Dataset(
        args.data_path,
        "train",
        ids=train_ids,
        los_thresh=los_threshold,
        static_only=True,
    )

    training_set.print_label_dist()

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

    validation_set.print_label_dist()

    x_val = []
    y_val = []
    for data in validation_set:
        x, y = data
        x_val.append(x[0])
        y_val.append(y[0])

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    if with_hp:
        params = [
            {
                "n_estimators": [100, 1000],
                "criterion": ["gini", "entropy"],
            }
        ]

        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=0, class_weight="balanced"),
            param_grid=params,
            scoring=["balanced_accuracy", "roc_auc", "accuracy"],
            refit="balanced_accuracy",
            cv=3,
        )
        print("Training via Grid Search..")

    else:
        print("Training a single RF model...")
        model = RandomForestClassifier(
            random_state=0, class_weight="balanced", criterion="entropy"
        )

    model.fit(x_train, y_train)

    if with_hp:
        print("Best params:", model.best_params_)
        print("Best score:", model.best_score_)

    print("Evaluating on validation data...")
    y_hat = model.predict(x_val)
    prob = model.predict_proba(x_val)[:, 1]

    acc = accuracy_score(y_val, y_hat)
    bacc = balanced_accuracy_score(y_val, y_hat)
    auc = roc_auc_score(y_val, prob)
    print("Predicting on validation...")
    print("Performance summary:", [acc, bacc, auc])
