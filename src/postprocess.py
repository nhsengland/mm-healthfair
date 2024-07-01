import argparse
import os
import pickle

import numpy as np
import polars as pl
import toml
from datasets import MIMIC4Dataset
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from utils.functions import load_pickle, read_from_txt

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
        "--train", default=None, help="List of ids to use for training."
    )
    parser.add_argument(
        "--val", default=None, help="List of ids to use for validation."
    )
    parser.add_argument(
        "--attribute",
        default="gender",
        type=str,
        help="The name of the protected characteristic. Defaults to 'gender'.",
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

    # Get training and validation ids
    train_ids = read_from_txt(args.train) if args.train is not None else None
    val_ids = read_from_txt(args.val) if args.val is not None else None

    # if loading from .ckpt (deep learning) set static_only to False else assume static only model (RF)
    model_type = "fusion" if os.path.splitext(args.model_path)[1] == ".ckpt" else "rf"
    static_only = True if model_type == "rf" else False

    train_set = MIMIC4Dataset(
        data_path,
        "train",
        ids=train_ids,
        los_thresh=los_threshold,
        static_only=static_only,
    )
    train_set.print_label_dist()

    # Get metadata containing sensitive features
    metadata = (
        pl.scan_csv(metadata_path)
        .filter(pl.col("hadm_id").is_in(list(map(int, train_ids))))
        .select(args.attribute)
        .collect()
    )
    if model_type == "rf":
        print("Loading dataset...")
        x_train = []
        y_train = []
        for data in train_set:
            x, y = data
            x_train.append(x[0])
            y_train.append(y[0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print("Training an unbiased model using Fairlearn's Threshold Optimiser...")
        unmitigated_model = load_pickle(args.model_path)

        model = ThresholdOptimizer(
            estimator=unmitigated_model,
            constraints="false_negative_rate_parity",
            objective="balanced_accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        )

        model.fit(x_train, y_train, sensitive_features=metadata)

        plot_threshold_optimizer(model)

        # Save unbiased model to disk
        log_dir = os.path.dirname(args.model_path)

        with open(
            os.path.join(log_dir, f"rf_mitigated_{args.attribute}.pkl"), "wb"
        ) as f:
            pickle.dump(model, f)

        print("Evaluating on validation data...")
        validation_set = MIMIC4Dataset(
            data_path,
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

        val_metadata = (
            pl.scan_csv(metadata_path)
            .filter(pl.col("hadm_id").is_in(list(map(int, val_ids))))
            .select(args.attribute)
            .collect()
        )

        y_hat = model.predict(x_val, sensitive_features=val_metadata)
        # prob = model.predict_proba(x_val)[:, 1]

        acc = accuracy_score(y_val, y_hat)
        bacc = balanced_accuracy_score(y_val, y_hat)
        cf = confusion_matrix(y_val, y_hat)
        print("Predicting on validation...")
        print("Performance summary:", [acc, bacc])
        print(cf)

    elif model_type == "fusion":
        # Use adversarial learning to debias model
        pass
