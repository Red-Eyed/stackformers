"""GEGLU feed-forward network with separate gate/value projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.feedforward.config import GEGLUConfig
    from stackformers.feedforward.packed_geglu import PackedGEGLU


class GEGLU(nn.Module):
    """GEGLU feed-forward network.

    GLU variant that gates with GELU instead of SiLU (SwiGLU). Empirically
    competitive with SwiGLU; slightly smoother gradient signal near zero.
    inner_dim = 2/3 * dim * mult so total params match a standard 4x GELU FFN.

    Paper: "GLU Variants Improve Transformers" — https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: GEGLUConfig) -> None:
        super().__init__()
        self.config = config.model_copy(deep=True)
        inner_dim = config.inner_dim

        self.w1 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w2 = nn.Linear(config.dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU(approximate="tanh")

        nn.init.normal_(self.w3.weight, std=0.02)

    @override
    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        gate = self.act(self.w1(x))
        hidden = gate * self.w2(x)
        output: Tensor = self.w3(self.dropout(hidden))
        return output

    def to_packed(self) -> PackedGEGLU:
        """Copy weights into packed gate/value projections before creating an optimizer."""
        from stackformers.feedforward.packed_geglu import PackedGEGLU

        return PackedGEGLU.from_unpacked(self)

    if TYPE_CHECKING:
        __call__ = forward
