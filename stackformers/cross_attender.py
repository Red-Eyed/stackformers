from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stackformers.attention.protocols import CrossAttn
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.sequence import PackedInput, SequenceInput


class CrossAttenderLayer(nn.Module):
    """Pre-norm layer: cross-attn → feed-forward. No self-attention."""

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.ff = ff
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        normed = x_input._replace(x=self.norm_cross(x_input.x))
        x = x_input.x + self.cross_attn(normed, ctx_input)
        x = x + self.ff(self.norm_ff(x))
        return x_input._replace(x=x)


class CrossAttenderStack(nn.Module):
    """Stack of CrossAttenderLayers with a final layer norm."""

    def __init__(
        self,
        layers: list[CrossAttenderLayer],
        final_norm: Norm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        for layer in self.layers:
            x_input = layer(x_input, ctx_input)
        return self.final_norm(x_input.x)


class PackedCrossAttenderLayer(nn.Module):
    """Pre-norm layer for packed variable-length sequences: cross-attn → feed-forward."""

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.ff = ff
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: PackedInput, ctx_input: PackedInput) -> PackedInput:
        normed = x_input._replace(x=self.norm_cross(x_input.x))
        x = x_input.x + self.cross_attn(normed, ctx_input)
        x = x + self.ff(self.norm_ff(x))
        return x_input._replace(x=x)


class PackedCrossAttenderStack(nn.Module):
    """Stack of PackedCrossAttenderLayers with a final layer norm."""

    def __init__(
        self,
        layers: list[PackedCrossAttenderLayer],
        final_norm: Norm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(self, x_input: PackedInput, ctx_input: PackedInput) -> Tensor:
        for layer in self.layers:
            x_input = layer(x_input, ctx_input)
        return self.final_norm(x_input.x)
