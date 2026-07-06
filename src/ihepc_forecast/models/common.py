from __future__ import annotations

import torch
from torch import nn


class ForecastModel(nn.Module):
    """三类直接多步预测模型的统一接口。"""

    def forward(
        self, x: torch.Tensor, future_calendar: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class LearnedPosition(nn.Module):
    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.shape[1]]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

