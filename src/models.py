# LSTM Model for time-series data embedding

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import bmm, nn, tanh


# nn.Modules
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, input_):
        output, (_, _) = self.lstm(input_)
        return output


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, (hidden, _) = self.lstm(packed_input)

        # Unpack the packed sequence
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Attention mechanism
        attention_weights = tanh(self.attention(padded_output))
        attention_scores = F.softmax(self.context_vector(attention_weights), dim=1)
        context_vector = bmm(attention_scores.transpose(1, 2), padded_output).squeeze(1)

        return context_vector, hidden


class AttentionLSTMWithMasking(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x, lengths):
        # Create a mask tensor based on missing values
        mask = ~(torch.isnan(x).any(dim=-1))

        # Pack the padded sequence with the mask
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, (hidden, _) = self.lstm(packed_input)

        # Unpack the packed sequence
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Apply the mask to the output
        padded_output = padded_output * mask.unsqueeze(-1)

        # Attention mechanism
        attention_weights = torch.tanh(self.attention(padded_output))
        attention_scores = F.softmax(self.context_vector(attention_weights), dim=1)
        context_vector = torch.bmm(
            attention_scores.transpose(1, 2), padded_output
        ).squeeze(1)

        return context_vector, hidden


class Gate(nn.Module):
    # Adapted from https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136 inspired by https://ieeexplore.ieee.org/document/9746536
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


# lightning.LightningModules


class MMModel(L.LightningModule):
    def __init__(
        self,
        st_input_dim=18,
        st_embed_dim=256,
        ts_input_dim=(9, 7),
        ts_embed_dim=256,
        num_ts=2,
        target_size=1,
        lr=0.1,
        fusion_method="concat",
        with_packed_sequences=False,
    ):
        super().__init__()
        self.num_ts = num_ts
        self.embed_timeseries = nn.ModuleList(
            [
                LSTM(
                    ts_input_dim[i],
                    ts_embed_dim,
                )
                for i in range(self.num_ts)
            ]
        )

        self.embed_static = nn.Linear(st_input_dim, st_embed_dim)

        self.fusion_method = fusion_method
        if self.fusion_method == "mag":
            self.fuse = Gate(st_embed_dim, *([ts_embed_dim] * self.num_ts), dropout=0.1)
            self.fc = nn.Linear(st_embed_dim, target_size)

        elif self.fusion_method == "concat":
            # embeddings must be same dim
            assert st_embed_dim == ts_embed_dim
            self.fc = nn.Linear(st_embed_dim * 2, target_size)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.with_packed_sequences = with_packed_sequences

    def prepare_batch(self, batch):
        if self.with_packed_sequences:
            s, d, l, y = batch  # static, dynamic, lengths, labels  # noqa: E741
        else:
            s, d, y = batch

        ts_embed = []
        for i in range(self.num_ts):
            if self.with_packed_sequences:
                packed_d = torch.nn.utils.rnn.pack_padded_sequence(
                    d[i], l[i], batch_first=True, enforce_sorted=False
                )
                embed = self.embed_timeseries[i](packed_d)
            else:
                embed = self.embed_timeseries[i](d[i])

            #  unpack if using packed sequences
            if self.with_packed_sequences:
                embed, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    embed, batch_first=True
                )

            ts_embed.append(embed[:, -1].unsqueeze(1))

        st_embed = self.embed_static(s)

        # Fuse time-series and static data
        if self.fusion_method == "concat":
            # use * to allow variable number of ts_embeddings
            # concat along feature dim
            out = torch.concat([st_embed, *ts_embed], dim=-1).squeeze()  # b x dim*2
        elif self.fusion_method == "mag":
            out = self.fuse(st_embed, *ts_embed)  # b x st_embed_dim

        # Parse through FC + sigmoid layer
        logits = self.fc(out)  # b x 1
        x_hat = F.sigmoid(logits)  # b x 1
        if len(x_hat.shape) < 2:  # noqa: PLR2004
            x_hat = x_hat.unsqueeze(0)
        return x_hat, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


class EmbedStatic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Use embedding for categorical static data (no ohe-needed)
        # Use dense layer for numerical static data
        # Concatenate outputs

    def forward(self):
        pass


class LitLSTM(L.LightningModule):
    """LSTM using time-series data only.

    Args:
        L (_type_): _description_
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
        if self.with_packed_sequences:
            _, d, l, y = batch  # static, dynamic, lengths, labels  # noqa: E741
            d = torch.nn.utils.rnn.pack_padded_sequence(
                d, l, batch_first=True, enforce_sorted=False
            )

        else:
            _, d, y = batch

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
        # training_step defines the train loop.
        # it is independent of forward
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
