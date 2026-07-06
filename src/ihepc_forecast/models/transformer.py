from __future__ import annotations

import torch
from torch import nn

from .common import ForecastModel, LearnedPosition


class TransformerForecaster(ForecastModel):
    """Vanilla encoder-decoder Transformer for multi-step forecasting.

    The design follows the forecasting protocol used by Informer and the
    THUML Time-Series-Library Transformer baseline.  The encoder receives the
    90-day multivariate history.  The decoder receives a short known-history
    label segment followed by zero placeholders for the unknown future, plus
    the known future calendar covariates.  Its final ``horizon`` tokens are
    projected to the target sequence.
    """

    def __init__(
        self,
        input_dim: int,
        input_days: int,
        horizon: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        decoder_layers: int = 1,
        dim_feedforward: int = 192,
        dropout: float = 0.15,
        label_days: int = 30,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.label_days = min(label_days, input_days)

        self.encoder_projection = nn.Linear(input_dim, d_model)
        self.encoder_position = LearnedPosition(input_days, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        decoder_length = self.label_days + horizon
        self.decoder_value_projection = nn.Linear(input_dim, d_model)
        self.decoder_calendar_projection = nn.Linear(4, d_model, bias=False)
        self.decoder_position = LearnedPosition(decoder_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_projection = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, future_calendar: torch.Tensor | None = None
    ) -> torch.Tensor:
        if future_calendar is None:
            future_calendar = x.new_zeros(x.shape[0], self.horizon, 4)
        if future_calendar.shape[1:] != (self.horizon, 4):
            raise ValueError(
                "future_calendar must have shape "
                f"[batch, {self.horizon}, 4], got {tuple(future_calendar.shape)}"
            )

        encoder_tokens = self.encoder_position(self.encoder_projection(x))
        memory = self.encoder(encoder_tokens)

        known_context = x[:, -self.label_days :]
        future_placeholders = x.new_zeros(x.shape[0], self.horizon, x.shape[-1])
        decoder_values = torch.cat([known_context, future_placeholders], dim=1)
        decoder_calendar = x.new_zeros(
            x.shape[0], self.label_days + self.horizon, 4
        )
        decoder_calendar[:, self.label_days :] = future_calendar
        decoder_tokens = self.decoder_position(
            self.decoder_value_projection(decoder_values)
            + self.decoder_calendar_projection(decoder_calendar)
        )

        length = decoder_tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        decoded = self.decoder(
            decoder_tokens,
            memory,
            tgt_mask=causal_mask,
        )
        return self.output_projection(decoded[:, -self.horizon :]).squeeze(-1)
