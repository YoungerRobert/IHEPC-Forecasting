from __future__ import annotations

import torch
from torch import nn

from .common import ForecastModel, LearnedPosition


class STLFormer(ForecastModel):
    """Spectral-Temporal Local Former.

    STL-Former extends the TL-style hybrid model with task-specific household
    power priors.  Its encoder has three complementary views:

    * local view: multi-scale temporal convolutions capture 3/7/15-day local
      patterns, including weekly habits;
    * spectral view: rFFT amplitudes summarize dominant periodic components in
      the 90-day history;
    * global view: self-attention models non-local dependencies over the
      enriched daily tokens, followed by an LSTM temporal re-reader.

    A future-calendar decoder then generates one token for every target day.
    The model is still trained end-to-end for each horizon.
    """

    def __init__(
        self,
        input_dim: int,
        input_days: int,
        horizon: int,
        d_model: int = 96,
        hidden_dim: int = 96,
        nhead: int = 4,
        transformer_layers: int = 2,
        decoder_layers: int = 1,
        dim_feedforward: int | None = None,
        dropout: float = 0.15,
        spectral_bins: int = 16,
        use_multiscale: bool = True,
        use_spectral: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.spectral_bins = spectral_bins
        self.use_multiscale = use_multiscale
        self.use_spectral = use_spectral
        dim_feedforward = dim_feedforward or d_model * 2

        self.input_projection = nn.Linear(input_dim, d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.conv15 = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7)
        self.local_norm = nn.LayerNorm(d_model)
        self.local_dropout = nn.Dropout(dropout)

        self.spectral_projection = nn.Sequential(
            nn.Linear(input_dim * spectral_bins, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.input_position = LearnedPosition(input_days, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.temporal_reader = nn.LSTM(
            d_model,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.memory_projection = nn.Linear(hidden_dim + d_model, d_model)

        self.calendar_projection = nn.Linear(4, d_model)
        self.horizon_position = LearnedPosition(horizon, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.calendar_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.context_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        if not self.use_multiscale:
            return projected
        channels = projected.transpose(1, 2)
        local = (
            self.conv3(channels)
            + self.conv7(channels)
            + self.conv15(channels)
        ).transpose(1, 2) / 3.0
        return self.local_dropout(self.local_norm(torch.relu(local) + projected))

    def _spectral_context(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_spectral:
            return x.new_zeros(x.shape[0], self.spectral_projection[-1].out_features)
        spectrum = torch.fft.rfft(x, dim=1).abs()
        bins = spectrum[:, 1 : self.spectral_bins + 1]
        if bins.shape[1] < self.spectral_bins:
            pad = x.new_zeros(x.shape[0], self.spectral_bins - bins.shape[1], x.shape[-1])
            bins = torch.cat([bins, pad], dim=1)
        # Normalize per sample so the branch focuses on relative periodic
        # strength rather than duplicating the standardized amplitude scale.
        bins = bins / bins.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return self.spectral_projection(bins.flatten(start_dim=1))

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

        local_tokens = self._local_tokens(x)
        spectral = self._spectral_context(x)
        enriched = local_tokens + spectral.unsqueeze(1)
        global_tokens = self.global_encoder(self.input_position(enriched))
        read_sequence, _ = self.temporal_reader(global_tokens)
        memory = self.memory_projection(torch.cat([read_sequence, global_tokens], dim=-1))

        queries = self.horizon_position(self.calendar_projection(future_calendar))
        decoded = self.calendar_decoder(queries, memory)
        context = self.context_head(memory.mean(dim=1)).unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        spectral_future = spectral.unsqueeze(1).expand(-1, self.horizon, -1)
        output = self.output_head(torch.cat([decoded, context, spectral_future], dim=-1))
        return output.squeeze(-1)
