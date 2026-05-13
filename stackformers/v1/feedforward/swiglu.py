from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from stackformers.v1.feedforward.config import FeedForwardConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Noam Shazeer, 2020).

    Uses two parallel gate matrices (W1, W2) and one output projection (W3).
    inner_dim = 2/3 * dim * mult so total params match a standard 4x GELU FFN.
    """

    def __init__(self, config: FeedForwardConfig) -> None:
        super().__init__()
        inner_dim = config.inner_dim

        self.w1 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w2 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.w3.weight, std=0.02)

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        gate = F.silu(self.w1(x))
        hidden = gate * self.w2(x)
        return self.w3(self.dropout(hidden))
