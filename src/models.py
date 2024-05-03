# LSTM Model for time-series data embedding

import torch
import torch.nn.functional as F
from torch import bmm, nn, tanh


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


# Example usage
# input_size = 10  # Dimensionality of input features
# hidden_size = 32  # Dimensionality of hidden state
# num_layers = 2  # Number of LSTM layers
# attention_size = 16  # Size of attention mechanism

# # Create an instance of the AttentionLSTMWithMasking model
# model = AttentionLSTMWithMasking(input_size, hidden_size, num_layers, attention_size)

# # Example input data and lengths (variable-length sequences)
# input_data = torch.randn(5, 10, input_size)  # Batch size x Sequence length x Input size
# # Introduce missing values
# input_data[2, 5:8, :] = float('nan')
# lengths = [10, 9, 5, 7, 6]  # Lengths of each sequence in the batch

# # Forward pass
# context_vector, hidden = model(input_data, lengths)
# print("Output shape:", context_vector.shape)  # Shape: (batch_size, hidden_size)

# Example usage
# input_size = 10  # Dimensionality of input features
# hidden_size = 32  # Dimensionality of hidden state
# num_layers = 2  # Number of LSTM layers
# attention_size = 16  # Size of attention mechanism

# # Create an instance of the AttentionLSTM model
# model = AttentionLSTM(input_size, hidden_size, num_layers, attention_size)

# # Example input data and lengths (variable-length sequences)
# input_data = torch.randn(5, 10, input_size)  # Batch size x Sequence length x Input size
# lengths = [10, 9, 8, 7, 6]  # Lengths of each sequence in the batch

# # Forward pass
# context_vector, hidden = model(input_data, lengths)
# print("Output shape:", context_vector.shape)  # Shape: (batch_size, hidden_size)
