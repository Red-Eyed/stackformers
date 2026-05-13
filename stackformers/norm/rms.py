from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean subtraction).

    Zhang & Sennrich, 2019 — https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        return F.normalize(x, dim=-1) * self.scale * self.g
