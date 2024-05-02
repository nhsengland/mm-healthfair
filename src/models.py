# LSTM Model for time-series data embedding

import torch.nn.functional as F
from torch import nn


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
