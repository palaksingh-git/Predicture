"""Next-digit model and simple baselines."""

import torch
from torch import nn


class NextDigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 32)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        _, hidden = self.gru(embedded)
        return self.classifier(hidden[-1])


def repeat_last_baseline(sequences):
    return sequences[:, -1]


def most_frequent_baseline(sequences):
    predictions = []
    for sequence in sequences:
        digits = sequence.tolist()
        counts = torch.bincount(sequence, minlength=10)
        maximum = counts.max()
        predictions.append(next(digit for digit in reversed(digits) if counts[digit] == maximum))
    return torch.tensor(predictions, dtype=torch.long, device=sequences.device)
