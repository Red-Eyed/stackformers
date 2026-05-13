from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.v1.attention.cross_attn import CrossAttention
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.sequence import SequenceInfo


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: causal self-attn → cross-attn → feed-forward."""

    def __init__(
        self,
        self_attn: SelfAttention,
        cross_attn: CrossAttention,
        ff: SwiGLU,
        norm_self: RMSNorm,
        norm_cross: RMSNorm,
        norm_ff: RMSNorm,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.norm_self = norm_self
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b s d"],
        tgt_seq_info: SequenceInfo,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]:
        x = x + self.self_attn(self.norm_self(x), tgt_seq_info)
        x = x + self.cross_attn(self.norm_cross(x), context, ctx_seq_info)
        x = x + self.ff(self.norm_ff(x))
        return x


class Decoder(nn.Module):
    """Stack of DecoderLayers with a final layer norm."""

    def __init__(
        self,
        layers: list[DecoderLayer],
        final_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b s d"],
        tgt_seq_info: SequenceInfo,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]:
        for layer in self.layers:
            x = layer(x, context, tgt_seq_info, ctx_seq_info)
        return self.final_norm(x)
