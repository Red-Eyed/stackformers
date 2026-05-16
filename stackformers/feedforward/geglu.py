from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from stackformers.feedforward.config import GEGLUConfig


class GEGLU(nn.Module):
    """GEGLU feed-forward network.

    GLU variant that gates with GELU instead of SiLU (SwiGLU). Empirically
    competitive with SwiGLU; slightly smoother gradient signal near zero.
    inner_dim = 2/3 * dim * mult so total params match a standard 4x GELU FFN.

    Paper: "GLU Variants Improve Transformers" — https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: GEGLUConfig) -> None:
        super().__init__()
        inner_dim = config.inner_dim

        self.w1 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w2 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.w3.weight, std=0.02)

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        gate = F.gelu(self.w1(x))
        hidden = gate * self.w2(x)
        return self.w3(self.dropout(hidden))
