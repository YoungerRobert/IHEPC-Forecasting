from __future__ import annotations

from .common import ForecastModel
from .lstm import LSTMForecaster
from .stl_former import STLFormer
from .transformer import TransformerForecaster


def build_model(
    name: str,
    input_dim: int,
    input_days: int,
    horizon: int,
    target_index: int = 0,
    d_model: int = 96,
    hidden_dim: int = 96,
    dropout: float = 0.15,
    use_multiscale: bool = True,
    use_spectral: bool = True,
) -> ForecastModel:
    key = name.lower().replace("-", "_")
    if key == "lstm":
        return LSTMForecaster(input_dim, horizon, hidden_dim, dropout=dropout)
    if key == "transformer":
        return TransformerForecaster(
            input_dim, input_days, horizon, d_model=d_model, dropout=dropout
        )
    if key in {"stl_former", "stlformer", "stl"}:
        return STLFormer(
            input_dim=input_dim,
            input_days=input_days,
            horizon=horizon,
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_multiscale=use_multiscale,
            use_spectral=use_spectral,
        )
    raise ValueError(f"未知模型: {name}")
