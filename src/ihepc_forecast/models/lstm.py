from __future__ import annotations

import torch
from torch import nn

from .common import ForecastModel


class LSTMForecaster(ForecastModel):
    """Vector-output LSTM baseline for direct multi-step forecasting.

    This follows the common household-power forecasting recipe: encode the
    complete look-back window, keep the final recurrent state, and use a small
    dense head to predict the full horizon in one operation.  The 90-day and
    365-day tasks therefore use independent output heads and independent
    training runs.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self, x: torch.Tensor, future_calendar: torch.Tensor | None = None
    ) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])
