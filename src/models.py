import csv
import os

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


# nn.Modules
class LSTM(nn.Module):
    """
    General-purpose LSTM module for sequence embeddings.

    Args:
        input_dim (int): Input feature dimension.
        embed_dim (int): Output embedding dimension.
        num_layers (int): Number of LSTM layers.
        hidden_dim (int): Hidden state dimension.
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Embedded output.
    """

    def __init__(self, input_dim, embed_dim, num_layers=1, hidden_dim=128, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.project = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_):
        _, (h_T, _) = self.lstm(input_)
        output = self.dropout(self.project(h_T[-1]))
        return self.relu(output)


class Gate(nn.Module):
    """
    Gated fusion module for combining multiple input embeddings.
    Adapted from https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136 inspired by https://arxiv.org/pdf/1908.05787.
    Args:
        inp1_size (int): Size of first input.
        inp2_size (int): Size of second input.
        inp3_size (int): Size of third input (optional).
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Fused output.
    """

    def __init__(self, inp1_size, inp2_size, inp3_size: int = 0, dropout: int = 0):
        super().__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc2 = nn.Linear(inp1_size + inp3_size, 1)
        self.fc3 = nn.Linear(inp2_size + inp3_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2, inp3=None):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        if inp3 is not None:
            w3 = torch.sigmoid(self.fc2(torch.cat([inp1, inp3], -1)))
            adjust = self.fc3(torch.cat([w2 * inp2, w3 * inp3], -1))
        else:
            # only need to adjust input 2
            adjust = self.fc3(w2 * inp2)

        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output)).squeeze()
        return output


### Adversarial debiasing component for main model using Gradient Reversal Layer
## Used to restrict the main model from learning sensitive attributes
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.

    Used to reverse gradients during backpropagation for adversarial debiasing.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    """
    Apply gradient reversal to the input tensor to enable maximisation of the adversarial objective function.

    Args:
        x (torch.Tensor): Input tensor.
        lambda_ (float): Scaling factor for gradient reversal.

    Returns:
        torch.Tensor: Output tensor with reversed gradients.
    """
    return GradientReversalFunction.apply(x, lambda_)


class MMModel(L.LightningModule):
    """
    Multimodal model object for fusion of static, timeseries, and notes data.

    Args:
        st_input_dim (int): Static input dimension.
        st_embed_dim (int): Static embedding dimension.
        ts_input_dim (tuple): Tuple of timeseries input dimensions.
        ts_embed_dim (int): Timeseries embedding dimension.
        nt_input_dim (int): Notes input dimension.
        nt_embed_dim (int): Notes embedding dimension.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        num_ts (int): Number of timeseries modalities.
        target_size (int): Output size.
        lr (float): Learning rate.
        fusion_method (str): Fusion method ("concat" or "mag").
        st_first (bool): If True, static features are fused first.
        modalities (list): List of modalities to use.
        with_packed_sequences (bool): If True, use packed sequences for timeseries.
        dataset (MIMIC4Dataset): Pass training dataset if using class weighting, else None.
        sensitive_attr_ids (list): Indices of sensitive attribute features from static data for adversarial debiasing.
        adv_lambda (float): Strength of adversarial penalty. No penalty is 0. Slight penalty is 0.1-0.2. Strong penalty is >=1.

    Returns:
        torch.Tensor: Model output.
    """

    def __init__(
        self,
        st_input_dim=18,
        st_embed_dim=64,
        ts_input_dim=(9, 7),
        ts_embed_dim=64,
        nt_input_dim=768,
        nt_embed_dim=64,
        num_layers=1,
        dropout=0.2,
        num_ts=2,
        target_size=1,
        lr=0.1,
        fusion_method="concat",
        st_first=True,
        modalities=None,
        with_packed_sequences=False,
        dataset=None,
        sensitive_attr_ids=None,  # list of indices in static features
        adv_lambda=0.0,  # strength of adversarial penalty
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modalities = set(modalities)
        self.with_static = "static" in self.modalities
        self.with_ts = "timeseries" in self.modalities
        self.with_notes = "notes" in self.modalities
        self.fusion_method = fusion_method
        self.st_first = st_first
        self.st_only = not self.with_ts and not self.with_notes
        self.ts_only = not self.with_static and not self.with_notes
        self.nt_only = not self.with_static and not self.with_ts
        self.st_ts = self.with_static and self.with_ts and not self.with_notes

        # Static embedding
        self.embed_static = (
            nn.Sequential(
                nn.Linear(st_input_dim, st_embed_dim // 2),
                nn.LayerNorm(st_embed_dim // 2),
                nn.Linear(st_embed_dim // 2, st_embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            if self.with_static
            else None
        )
        st_embed_dim = st_embed_dim if self.with_static else 0

        # Time-series embedding
        self.embed_timeseries = (
            nn.ModuleList(
                [
                    LSTM(
                        ts_input_dim[i],
                        ts_embed_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                    for i in range(num_ts)
                ]
            )
            if self.with_ts
            else None
        )
        ts_embed_dim = ts_embed_dim if self.with_ts else 0

        # Notes embedding
        self.embed_notes = (
            nn.Sequential(
                nn.Linear(nt_input_dim, nt_embed_dim),
                nn.LayerNorm(nt_embed_dim),
                nn.ReLU(),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=nt_embed_dim,
                        nhead=8,
                        dim_feedforward=256,
                        dropout=dropout,
                        activation="relu",
                        batch_first=True,
                    ),
                    num_layers=2,
                ),
                nn.Dropout(dropout),
            )
            if self.with_notes
            else None
        )
        nt_embed_dim = nt_embed_dim if self.with_notes else 0

        # Fusion and final layer
        if fusion_method == "mag" and self.with_ts and self.with_static:
            self.fuse = Gate(
                st_embed_dim if st_first else ts_embed_dim,
                *([ts_embed_dim] * num_ts if st_first else [st_embed_dim]),
                dropout=dropout,
            )
            self.fc = nn.Linear(st_embed_dim if st_first else ts_embed_dim, target_size)
        elif fusion_method == "concat":
            total_dim = st_embed_dim + (num_ts * ts_embed_dim) + nt_embed_dim
            self.fc = nn.Linear(total_dim, target_size)
        else:
            embed_dim = st_embed_dim or (num_ts * ts_embed_dim) or nt_embed_dim
            self.fc = nn.Linear(embed_dim, target_size)

        # Loss function with class weights
        pos_weight = self.compute_class_weights(dataset) if dataset else None
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Metrics
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.ap = torchmetrics.AveragePrecision(task="binary")
        self.with_packed_sequences = with_packed_sequences
        self.sensitive_attr_ids = sensitive_attr_ids or []
        self.adv_lambda = adv_lambda

        # Adversarial heads for each sensitive attribute (binary classification)
        self._init_adversarial_heads(
            st_embed_dim, ts_embed_dim, nt_embed_dim, num_ts, fusion_method
        )

    def _init_adversarial_heads(
        self, st_embed_dim, ts_embed_dim, nt_embed_dim, num_ts, fusion_method
    ):
        """
        Initialize adversarial classifier heads for sensitive attributes.

        Args:
            st_embed_dim (int): Static embedding dimension.
            ts_embed_dim (int): Timeseries embedding dimension.
            nt_embed_dim (int): Notes embedding dimension.
            num_ts (int): Number of timeseries modalities.
            fusion_method (str): Fusion method.
        """
        ### Adversarial classifier targeting sensitive attributes
        if self.sensitive_attr_ids:
            adv_in_dim = st_embed_dim if self.with_static else 0
            if fusion_method == "concat":
                adv_in_dim = st_embed_dim + (num_ts * ts_embed_dim) + nt_embed_dim
            elif fusion_method == "mag" and self.with_ts and self.with_static:
                adv_in_dim = st_embed_dim if self.st_first else ts_embed_dim
            elif self.with_static:
                adv_in_dim = st_embed_dim
            elif self.with_ts:
                adv_in_dim = num_ts * ts_embed_dim
            elif self.with_notes:
                adv_in_dim = nt_embed_dim
            self.adv_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(adv_in_dim, adv_in_dim // 2),
                        nn.ReLU(),
                        nn.Linear(adv_in_dim // 2, 1),
                    )
                    for _ in self.sensitive_attr_ids
                ]
            )
            self.adv_criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.adv_heads = None

    def prepare_batch(self, batch):
        """
        Prepare input batch for forward pass, handling all modalities.

        Args:
            batch: Input batch from DataLoader.

        Returns:
            tuple: Model output, labels, and optionally embeddings and static data for adversarial loss.
        """
        ### Unpack batch based on the available modalities
        if self.st_only:
            s, y, d, lengths, n = batch[0], batch[1], None, None, None
        elif self.st_ts or self.ts_only:
            s, y, d, lengths, n = batch[0], batch[1], batch[2], batch[3], None
        else:
            s, y, d, lengths, n = batch[0], batch[1], batch[2], batch[3], batch[4]

        st_embed = self.embed_static(s) if self.with_static else None
        ts_embed = (
            [
                self.embed_timeseries[i](
                    torch.nn.utils.rnn.pack_padded_sequence(
                        d[i], lengths[i], batch_first=True, enforce_sorted=False
                    )
                    if self.with_packed_sequences
                    else d[i]
                ).unsqueeze(1)
                for i in range(self.hparams.num_ts)
            ]
            if self.with_ts
            else None
        )
        nt_embed = self.embed_notes(n) if self.with_notes else None

        # Fuse embeddings
        if self.fusion_method == "concat":
            embeddings = [
                e for e in [st_embed, *(ts_embed or []), nt_embed] if e is not None
            ]
            out = torch.concat(embeddings, dim=-1).squeeze()
        elif self.fusion_method == "mag" and self.with_ts:
            out = (
                self.fuse(st_embed, *ts_embed)
                if self.st_first
                else self.fuse(*ts_embed, st_embed)
            )
        elif self.st_only:
            out = st_embed.squeeze()
        elif self.ts_only:
            out = torch.cat(ts_embed, dim=-1).squeeze()
        elif self.nt_only:
            out = nt_embed.squeeze()

        x_hat = self.fc(out)
        # Return embeddings for adversarial loss if needed
        return (
            (x_hat.unsqueeze(0) if len(x_hat.shape) == 1 else x_hat, y, out, s)
            if self.adv_heads
            else (x_hat.unsqueeze(0) if len(x_hat.shape) == 1 else x_hat, y)
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        # training_step defines the train loop.
        # it is independent of forward
        if self.adv_heads:
            x_hat, y, rep, s = self.prepare_batch(batch)
        else:
            x_hat, y = self.prepare_batch(batch)
        y_hat = torch.sigmoid(x_hat)  # prob
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(y_hat, y)
        auc = self.auc(y_hat, y)
        f1 = self.f1(y_hat, y)
        ap = self.ap(y_hat, y.long())

        # Adversarial loss
        if self.adv_heads and self.adv_lambda > 0:
            adv_loss = 0
            # Access sensitive attributes from the unpacked static data 's'
            if s is not None:
                for i, idx in enumerate(self.sensitive_attr_ids):
                    # Ensure index 'idx' is within the bounds of the static data tensor
                    if idx < s.size(-1):
                        # Assuming s is batch_size x static_features
                        sensitive_label = s[:, :, idx].float().view(-1, 1)
                        adv_input = grad_reverse(rep, self.adv_lambda)
                        adv_pred = self.adv_heads[i](adv_input)
                        adv_loss += self.adv_criterion(adv_pred, sensitive_label)

                if len(self.sensitive_attr_ids) > 0:
                    adv_loss = adv_loss / len(self.sensitive_attr_ids)
                    loss = loss + self.adv_lambda * adv_loss
                    self.log(
                        "adv_loss",
                        adv_loss,
                        prog_bar=True,
                        on_epoch=True,
                        on_step=False,
                        batch_size=len(y),
                    )
                else:
                    # No sensitive attributes configured, no adversarial loss
                    pass
            else:
                # print("Warning: Adversarial heads defined but static data 's' is None.")
                pass  # Adversarial heads are defined but static data is not available

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_acc",
            accuracy,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_auc",
            auc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_f1",
            f1,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_ap",
            ap,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        # Do not penalize adversarial loss in validation
        if self.adv_heads:
            x_hat, y, _, _ = self.prepare_batch(
                batch
            )  # Still unpack all returned values
        else:
            x_hat, y = self.prepare_batch(batch)
        y_hat = torch.sigmoid(x_hat)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(y_hat, y)
        auc = self.auc(y_hat, y)
        f1 = self.f1(y_hat, y)
        ap = self.ap(y_hat, y.long())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        self.log("val_auc", auc, prog_bar=True, batch_size=len(y))
        self.log("val_f1", f1, prog_bar=True, batch_size=len(y))
        self.log("val_ap", ap, prog_bar=True, batch_size=len(y))

    def predict_step(self, batch):
        """
        Prediction step for one batch.

        Args:
            batch: Input batch.

        Returns:
            tuple: (predictions, labels)
        """
        if self.adv_heads:
            x_hat, y, _, _ = self.prepare_batch(batch)
        else:
            x_hat, y = self.prepare_batch(batch)
        y_hat = torch.sigmoid(x_hat)
        return y_hat, y

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            tuple: (list of optimizers, list of schedulers)
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return [optimizer], [
            {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        ]

    def compute_class_weights(self, dataset):
        """
        Compute class weights for binary classification.

        Args:
            dataset: The dataset containing labels.

        Returns:
            torch.Tensor: Class weights for the positive class.
        """
        labels = []
        for i in range(len(dataset)):  # Assuming dataset returns (data, label, ...)
            labels.append(
                dataset[i][1].item()
                if isinstance(dataset[i][1], torch.Tensor)
                else dataset[i][1]
            )

        total_samples = len(labels)
        positive_samples = sum(labels)
        negative_samples = total_samples - positive_samples

        # Compute weights
        pos_weight = negative_samples / positive_samples
        print(f"Adjusting positive class weight to: {round(pos_weight, 3)}")
        return torch.tensor(pos_weight, dtype=torch.float32)


class LitLSTM(L.LightningModule):
    """
    LSTM model for time-series data only.

    Args:
        ts_input_dim (int): Timeseries input dimension.
        lstm_embed_dim (int): LSTM embedding dimension.
        target_size (int): Output size.
        lr (float): Learning rate.
        with_packed_sequences (bool): If True, use packed sequences.

    Returns:
        torch.Tensor: Model output.
    """

    def __init__(
        self,
        ts_input_dim,
        lstm_embed_dim,
        target_size,
        lr=0.1,
        with_packed_sequences=False,
    ):
        super().__init__()
        self.embed_timeseries = LSTM(
            ts_input_dim,
            lstm_embed_dim,
            target_size,
            with_packed_sequences=with_packed_sequences,
        )
        self.fc = nn.Linear(lstm_embed_dim, target_size)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.with_packed_sequences = with_packed_sequences

    def prepare_batch(self, batch):
        """
        Prepare input batch for LSTM model.

        Args:
            batch: Input batch from DataLoader.

        Returns:
            tuple: (predictions, labels)
        """
        if self.with_packed_sequences:
            _, y, d, l = batch  # static, dynamic, lengths, labels
            d = torch.nn.utils.rnn.pack_padded_sequence(
                d, l, batch_first=True, enforce_sorted=False
            )

        else:
            _, y, d = batch

        ts_embed = self.embed_timeseries(d)

        # unpack if using packed sequences
        if self.with_packed_sequences:
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                ts_embed, batch_first=True
            )

        # [:, -1] for hidden state at the last time step
        logits = self.fc(lstm_out[:, -1])
        x_hat = F.sigmoid(logits)
        return x_hat, y

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        # training_step defines the train loop.
        # it is independent of forward
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for LSTM model.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SaveLossesCallback(Callback):
    """
    Learner callback to save train/validation losses to a CSV file.
    """

    def __init__(self, log_dir="logs", save_every_n_epochs=5):
        """
        Callback to save train/validation losses to a CSV file every n epochs.

        Args:
            log_dir (str): Directory to save the logs.
            save_every_n_epochs (int): Interval (in epochs) to save the losses.
        """
        self.log_dir = log_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = os.path.join(self.log_dir, "losses.csv")

        # Initialize the CSV file with headers if it doesn't exist
        with open(self.csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.

        Returns:
            None
        """
        # Save losses every n epochs
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            train_loss = trainer.callback_metrics.get("train_loss", None)
            val_loss = trainer.callback_metrics.get("val_loss", None)

            # Append the losses to the CSV file
            with open(self.csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        trainer.current_epoch + 1,
                        train_loss.item() if train_loss is not None else None,
                        val_loss.item() if val_loss is not None else None,
                    ]
                )

            print(f"Saved losses to {self.csv_file}")
