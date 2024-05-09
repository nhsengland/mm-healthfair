# LSTM Model for time-series data embedding

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import bmm, nn, tanh


# nn.Modules
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_size)

    def forward(self, input_):
        lstm_out, (h, c) = self.lstm(input_)
        logits = self.fc(lstm_out[:, -1])
        scores = F.sigmoid(logits)
        return scores


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


# lightning.LightningModules
class LitLSTM(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, target_size, lr=0.1):
        super().__init__()
        self.net = LSTM(input_dim, hidden_dim, target_size)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _, y = batch
        x_hat = self.net(x)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x_hat = self.net(x)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(x))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
