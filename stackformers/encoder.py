from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.layers import TransformerLayer
from stackformers.norm.protocols import Norm
from stackformers.sequence import SequenceInfo


class Encoder(nn.Module):
    """Stack of TransformerLayers with a final layer norm."""

    def __init__(
        self,
        layers: list[TransformerLayer],
        final_norm: Norm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        seq_info: SequenceInfo,
    ) -> Float[Tensor, "b n d"]:
        for layer in self.layers:
            x = layer(x, seq_info)
        return self.final_norm(x)
