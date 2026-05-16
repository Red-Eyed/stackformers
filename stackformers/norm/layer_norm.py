from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.norm.config import LayerNormConfig


class LayerNorm(nn.Module):
    """Layer Normalization — normalises mean and variance across the feature dimension.

    Adds learnable affine parameters (weight and bias) after normalisation.
    Unlike RMSNorm, this subtracts the mean before scaling, making outputs
    zero-centred by construction.

    Paper: Ba et al., 2016 — https://arxiv.org/abs/1607.06450
    """

    def __init__(self, config: LayerNormConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.dim, eps=config.eps)

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        return self.norm(x)
