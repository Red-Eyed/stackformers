from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stackformers.attention.protocols import CrossAttn, SelfAttn
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.sequence import SequenceInput


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: causal self-attn → cross-attn → feed-forward."""

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_self: Norm,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.norm_self = norm_self
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        normed_self = x_input._replace(x=self.norm_self(x_input.x))
        x = x_input.x + self.self_attn(normed_self)
        x_input = x_input._replace(x=x)
        normed_cross = x_input._replace(x=self.norm_cross(x_input.x))
        x = x_input.x + self.cross_attn(normed_cross, ctx_input)
        x = x + self.ff(self.norm_ff(x))
        return x_input._replace(x=x)


class Decoder(nn.Module):
    """Stack of DecoderLayers with a final layer norm."""

    def __init__(
        self,
        layers: list[DecoderLayer],
        final_norm: Norm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        for layer in self.layers:
            x_input = layer(x_input, ctx_input)
        return self.final_norm(x_input.x)
