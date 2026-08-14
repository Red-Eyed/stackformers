"""Standard GELU feed-forward network."""

from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.feedforward.config import GELUConfig


class GELUFFN(nn.Module):
    """Apply a bias-free, two-projection GELU feed-forward transformation."""

    def __init__(self, config: GELUConfig) -> None:
        """Build the projections and activation described by ``config``."""
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.inner_dim, bias=False)
        self.w2 = nn.Linear(config.inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        """Transform token embeddings without changing their shape."""
        hidden = self.act(self.w1(x))
        return self.w2(self.dropout(hidden))
