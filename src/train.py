import argparse
import glob
import os
import sys

import lightning as L
import numpy as np
import polars as pl
import toml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset
from models import MMModel, SaveLossesCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal learning pipeline.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data dictionary generated from prepare_data.py.",
        default="../outputs/processed_data/mmfair_feat.pkl",
    )
    parser.add_argument(
        "--col_path",
        "-p",
        type=str,
        help="Path to the pickled column dictionary generated from prepare_data.py.",
        default="../outputs/processed_data/mmfair_cols.pkl",
    )
    parser.add_argument(
        "--ids_path",
        "-i",
        type=str,
        help="Directory containing train/val/test ids.",
        default="../outputs/processed_data",
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
        "--cpu", action="store_true", help="Whether to use cpu. Defaults to gpu"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample subjects for testing. Training set will equal sample, validation set will be 1/5th of sample.",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Whether to use class weights for training. Defaults to False.",
    )
    parser.add_argument(
        "--use_debiasing",
        action="store_true",
        help="Whether to use adversarial debiasing to penalize the model's decisions if influenced by sensitive attributes. Target attribute indices in dictionary can be defined in dbindices within targets.toml. Defaults to False.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="nhs-mm-healthfair",
        help="Name of project, used for wandb logging.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb for logging. Defaults to False",
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    targets = toml.load(args.targets)
    device = "gpu" if not args.cpu else "cpu"
    use_wandb = args.wandb

    ### Model config
    batch_size = config["data"]["batch_size"]
    n_epochs = config["train"]["epochs"]
    lr = config["train"]["learning_rate"]
    es = config["train"]["early_stopping"]
    num_workers = config["data"]["num_workers"]
    fusion_method = config["model"]["fusion_method"]
    ### Metadata for adversarial debiasing
    debiasing = args.use_debiasing
    adv_lambda = config["train"]["adv_lambda"] if debiasing else 0.0
    db_ids = targets["attributes"]["dbindices"] if debiasing else None
    # overrides to True if not using mag fusion method
    ### Modality-specific config
    st_first = config["model"]["st_first"] if fusion_method == "mag" else True
    modalities = config["data"]["modalities"]
    with_ts = True if "timeseries" in modalities else False
    static_only = True if (len(modalities) == 1) and ("static" in modalities) else False
    with_notes = True  ### Text modality is required to construct the data batches but is left out when not used for training.
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    if args.outcome not in outcomes:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    if fusion_method != "None" and len(modalities) == 1:
        print("Fusion method is not None, but only one modality is provided. Exiting..")
        sys.exit()

    outcome_idx = outcomes.index(args.outcome)
    print("------------------------------------------")
    print("MMHealthFair: Multimodal learning pipeline")
    print(
        f'Creating multimodal learning pipeline for outcome "{outcomes_disp[outcome_idx]}"'
    )
    print(f"Modalities used: {modalities}")
    print(f"Fusion method: {fusion_method}")
    print(f"With adversarial debiasing: {debiasing}")
    print("------------------------------------------")
    L.seed_everything(0)

    # Get training and validation ids
    if (
        len(
            glob.glob(
                os.path.join(args.ids_path, "training_ids_" + args.outcome + ".csv")
            )
        )
        == 0
    ):
        print(f"No training ids found for outcome {args.outcome}. Exiting..")
        sys.exit()

    if (
        len(
            glob.glob(
                os.path.join(args.ids_path, "validation_ids_" + args.outcome + ".csv")
            )
        )
        == 0
    ):
        print(f"No validation ids found for outcome {args.outcome}. Exiting..")
        sys.exit()

    train_ids = (
        pl.read_csv(
            os.path.join(args.ids_path, "training_ids_" + args.outcome + ".csv")
        )
        .select("subject_id")
        .to_numpy()
        .flatten()
    )
    val_ids = (
        pl.read_csv(
            os.path.join(args.ids_path, "validation_ids_" + args.outcome + ".csv")
        )
        .select("subject_id")
        .to_numpy()
        .flatten()
    )

    if args.sample is not None:
        ### Randomly sample train_ids
        print(
            f"Using random sample of {args.sample} subjects for training and validation."
        )
        train_ids = np.random.choice(train_ids, args.sample, replace=False)
        val_ids = np.random.choice(val_ids, args.sample // 5, replace=False)
        args.outcome = args.outcome + "_sample"

    training_set = MIMIC4Dataset(
        args.data_path,
        args.col_path,
        "train",
        ids=train_ids,
        static_only=static_only,
        with_notes=with_notes,
    )
    training_set.print_label_dist()
    # training_set.__getitem__(0)
    # sys.exit()

    n_static_features = training_set.get_feature_dim()  # add -1 if dropping label col

    if not static_only:
        n_dynamic_features = (
            training_set.get_feature_dim("dynamic_0"),
            training_set.get_feature_dim("dynamic_1"),
        )
        # print(n_dynamic_features)
        # print(n_static_features)
    else:
        n_dynamic_features = (None, None)

    training_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True if num_workers > 0 else False,
    )

    validation_set = MIMIC4Dataset(
        args.data_path,
        args.col_path,
        "val",
        ids=val_ids,
        static_only=static_only,
        with_notes=with_notes,
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True if num_workers > 0 else False,
    )
    validation_set.print_label_dist()

    model = MMModel(
        st_input_dim=n_static_features,
        ts_input_dim=n_dynamic_features,
        with_packed_sequences=True if not static_only else False,
        fusion_method=fusion_method,
        modalities=modalities,
        st_first=st_first,
        dataset=training_set
        if args.use_class_weights
        else None,  # Pass in dataset for adjusting class weights
        sensitive_attr_ids=db_ids,
        adv_lambda=adv_lambda,
    )

    # trainer
    if use_wandb:
        logger = WandbLogger(
            log_model=True,
            project=args.project,
            save_dir="logs",
        )
        # store config args
        logger.experiment.config.update(config)
    else:
        logger = CSVLogger("logs")

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=es)

    mod_str = "_".join(modalities)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename=f"{args.outcome}_{fusion_method}_{mod_str}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    save_losses_callback = SaveLossesCallback(
        log_dir=f"logs/{args.outcome}_{fusion_method}_{mod_str}/", save_every_n_epochs=5
    )

    trainer = L.Trainer(
        max_epochs=n_epochs,
        log_every_n_steps=5,
        logger=logger,
        accelerator=device,
        callbacks=[early_stop, checkpoint, lr_monitor, save_losses_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Completed multimodal training.")
