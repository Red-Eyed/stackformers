"""HardSwish-gated feed-forward network."""

from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.feedforward.config import HardSwishGLUConfig


class HardSwishGLU(nn.Module):
    """Parameter-matched GLU using a deployment-friendly HardSwish gate.

    HardSwish approximates the SiLU gate used by SwiGLU with a piecewise-linear
    function. Keeping this as a distinct type preserves SwiGLU's mathematical
    contract and makes serialized model configurations unambiguous.
    """

    def __init__(self, config: HardSwishGLUConfig) -> None:
        """Build three bias-free projections at the configured gated width."""
        super().__init__()
        inner_dim = config.inner_dim

        self.w1 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w2 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.Hardswish()

        nn.init.normal_(self.w3.weight, std=0.02)

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        """Gate one projection with HardSwish and return the original model width."""
        gate = self.act(self.w1(x))
        hidden = gate * self.w2(x)
        return self.w3(self.dropout(hidden))
