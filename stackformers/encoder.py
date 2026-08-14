"""Encoder stack for interchangeable transformer-layer topologies."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor

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

    def forward(self, input: SequenceInput) -> Tensor:
        for layer in self.layers:
            input = layer(input)
        return self.final_norm(input.x)
