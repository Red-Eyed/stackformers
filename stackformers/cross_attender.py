from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.attention.protocols import CrossAttn
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.sequence import SequenceInfo


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

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b s d"],
        x_seq_info: SequenceInfo | None = None,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]:
        x = x + self.cross_attn(self.norm_cross(x), context, x_seq_info, ctx_seq_info)
        x = x + self.ff(self.norm_ff(x))
        return x


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

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b s d"],
        x_seq_info: SequenceInfo | None = None,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]:
        for layer in self.layers:
            x = layer(x, context, x_seq_info, ctx_seq_info)
        return self.final_norm(x)
