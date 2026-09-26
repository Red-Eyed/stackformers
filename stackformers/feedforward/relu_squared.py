from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.feedforward.config import ReluSquaredConfig


class ReluSquaredFF(nn.Module):
    """ReLU² feed-forward network.

    Standard two-matrix FFN (no gating) with ReLU² activation.
    inner_dim = dim * mult; parameter count matches a standard mult-x FFN.

    Paper: "Primer: Searching for Efficient Transformers" — https://arxiv.org/abs/2109.08668
    """

    def __init__(self, config: ReluSquaredConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.inner_dim, bias=False)
        self.w2 = nn.Linear(config.inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.ReLU()

    @override
    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        h = self.act(self.w1(x))
        output: Tensor = self.w2(self.dropout(h.square()))
        return output

    if TYPE_CHECKING:
        __call__ = forward
