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
        "subjects_root_dir",
        type=str,
        help="Path to the subject-level data",
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

    # Create subject-level training and validation

    # now use torch.nn.utils.rnn.pack_padded_sequence() to pack according to the length
    # events = torch.nn.utils.rnn.pack_padded_sequence(
    #     events, timeseries_lengths, batch_first=True, enforce_sorted=False
    # )

    training_set = MIMIC4Dataset(
        "train", args.subjects_root_dir, los_thresh=los_threshold
    )
    training_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateTimeSeries(),
    )

    validation_set = MIMIC4Dataset(
        "val", args.subjects_root_dir, los_thresh=los_threshold
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateTimeSeries(),
    )

    # model = LitLSTM(input_dim=16, hidden_dim=256, target_size=1, lr=lr, with_packed_sequences=True)
    model = MMModel(with_packed_sequences=True, fusion_method=fusion_method)

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
