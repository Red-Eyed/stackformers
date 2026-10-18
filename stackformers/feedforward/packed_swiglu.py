"""Opt-in SwiGLU variant with packed shared-input projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers._projection_packing import load_packed_state, load_unpacked_state
from stackformers.feedforward.config import PackedSwiGLUConfig, SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU

if TYPE_CHECKING:
    from jaxtyping import Float


_SEPARATE = ("w1.weight", "w2.weight")
_PACKED = "gate_value.weight"


class PackedSwiGLU(nn.Module):
    """Share the original equation while packing input projection weights.

    Use ``from_unpacked`` to preserve existing weights and ``unpack`` to restore
    the original checkpoint layout. Convert before creating an optimizer.
    """

    def __init__(self, config: PackedSwiGLUConfig) -> None:
        """Build the input projections and shared SwiGLU output path."""
        super().__init__()
        self.config = config.model_copy(deep=True)
        inner_dim = config.inner_dim

        self.gate_value = nn.Linear(config.dim, 2 * inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.SiLU()

        nn.init.normal_(self.w3.weight, std=0.02)

    @classmethod
    def from_unpacked(cls, source: SwiGLU) -> PackedSwiGLU:
        """Copy an ordinary module without retaining or changing its parameters."""
        config = PackedSwiGLUConfig.model_validate(source.config.model_dump(exclude={"kind"}))
        with torch.device("meta"):
            target = cls(config)
        load_packed_state(source, target, _SEPARATE, _PACKED)
        return target

    def unpack(self) -> SwiGLU:
        """Return an independent module compatible with ordinary checkpoints."""
        with torch.device("meta"):
            target = SwiGLU(SwiGLUConfig.model_validate(self.config.model_dump(exclude={"kind"})))
        load_unpacked_state(self, target, _PACKED, _SEPARATE)
        return target

    @override
    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        """Apply the shared SwiGLU equation to packed input projections."""
        projected: Tensor = self.gate_value(x)
        projected_gate, value = projected.chunk(2, dim=-1)
        gate = self.act(projected_gate)
        hidden = gate * value
        output: Tensor = self.w3(self.dropout(hidden))
        return output

    if TYPE_CHECKING:
        __call__ = forward
