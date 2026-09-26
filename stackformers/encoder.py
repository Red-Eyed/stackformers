"""Encoder stack for interchangeable transformer-layer topologies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stackformers.layers import TransformerLayerBase
    from stackformers.norm.protocols import Norm
    from stackformers.sequence import SequenceInput


class Encoder(nn.Module):
    """Stack of TransformerLayers with a final layer norm."""

    def __init__(
        self,
        layers: Sequence[TransformerLayerBase],
        final_norm: Norm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    @override
    def forward(self, input: SequenceInput) -> Tensor:
        for layer in self.layers:
            input = layer(input)
        return self.final_norm(input.x)

    if TYPE_CHECKING:
        __call__ = forward
