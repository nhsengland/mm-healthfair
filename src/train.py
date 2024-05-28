import argparse

import lightning as L
import toml
from datasets import CollateTimeSeries, MIMIC4Dataset
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from models import MMModel
from torch.utils.data import DataLoader

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
    parser.add_argument("--ids", nargs="*", default=None, help="List of ids to use")
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
    los_threshold = config["threshold"]
    fusion_method = config["fusion_method"]
    exp_name = config["train"]["experiment_name"]

    L.seed_everything(0)

    # TODO: Create subject-level training and validation splits

    training_set = MIMIC4Dataset(args.data_path, "train", los_thresh=los_threshold)

    n_static_features = (
        training_set.get_feature_dim() - 1
    )  # -1 since extracting label from static data and dropping los column
    n_dynamic_features = (
        training_set.get_feature_dim("dynamic_0"),
        training_set.get_feature_dim("dynamic_1"),
    )

    training_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateTimeSeries(),
    )

    validation_set = MIMIC4Dataset(args.data_path, "val", los_thresh=los_threshold)
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateTimeSeries(),
    )

    model = MMModel(
        st_input_dim=n_static_features,
        ts_input_dim=n_dynamic_features,
        with_packed_sequences=True,
        fusion_method=fusion_method,
    )

    # trainer
    if use_wandb:
        logger = WandbLogger(
            log_model=True, project="nhs-mm-healthfair", save_dir="logs"
        )
    else:
        logger = CSVLogger("logs")

    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=n_epochs,
        log_every_n_steps=10,
        logger=logger,
        accelerator=device,
    )
    trainer.fit(
        model=model,
        train_dataloaders=training_dataloader,
        val_dataloaders=val_dataloader,
    )
