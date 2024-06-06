import argparse

import lightning as L
import toml
from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from models import MMModel
from torch.utils.data import DataLoader
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
    parser.add_argument(
        "--train", nargs="?", default=None, help="List of ids to use for training."
    )
    parser.add_argument(
        "--val", nargs="?", default=None, help="List of ids to use for validation."
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb for logging. Defaults to False",
    )
    args = parser.parse_args()

    config = toml.load(args.config)
    device = "gpu" if not args.cpu else "cpu"
    use_wandb = args.wandb

    batch_size = config["data"]["batch_size"]
    n_epochs = config["train"]["epochs"]
    lr = config["train"]["learning_rate"]
    num_workers = config["data"]["num_workers"]
    los_threshold = config["model"]["threshold"]
    fusion_method = (
        config["model"]["fusion_method"] if config["model"]["fusion_method"] else None
    )
    modalities = config["data"]["modalities"]
    static_only = True if (len(modalities) == 1) and ("static" in modalities) else False
    with_notes = True if "notes" in modalities else False

    L.seed_everything(0)

    # Get training and validation ids
    train_ids = read_from_txt(args.train) if args.train is not None else None
    val_ids = read_from_txt(args.val) if args.val is not None else None

    training_set = MIMIC4Dataset(
        args.data_path,
        "train",
        ids=train_ids,
        los_thresh=los_threshold,
        static_only=static_only,
    )

    training_set.print_label_dist()

    n_static_features = (
        training_set.get_feature_dim() - 1
    )  # -1 since extracting label from static data and dropping los column

    if not static_only:
        n_dynamic_features = (
            training_set.get_feature_dim("dynamic_0"),
            training_set.get_feature_dim("dynamic_1"),
        )
    else:
        n_dynamic_features = (None, None)

    training_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True,
    )

    validation_set = MIMIC4Dataset(
        args.data_path,
        "val",
        ids=val_ids,
        los_thresh=los_threshold,
        static_only=static_only,
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True,
    )

    model = MMModel(
        st_input_dim=n_static_features,
        ts_input_dim=n_dynamic_features,
        with_packed_sequences=True if not static_only else False,
        fusion_method=fusion_method,
    )

    # trainer
    if use_wandb:
        logger = WandbLogger(
            log_model=True,
            project="nhs-mm-healthfair",
            save_dir="logs",
        )
        # store config args
        logger.experiment.config.update(config)
    else:
        logger = CSVLogger("logs")

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=n_epochs,
        log_every_n_steps=50,
        logger=logger,
        accelerator=device,
        callbacks=[early_stop, checkpoint, lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dataloader,
        val_dataloaders=val_dataloader,
    )
