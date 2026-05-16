from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import SequenceInput


class TransformerDecoderConfig(BaseModel):
    self_attn: SelfAttentionConfig
    cross_attn: CrossAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig  # applies to self-attention only
    num_layers: int = Field(gt=0)


def plain_decoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerDecoderConfig:
    """Causal self-attn + cross-attn decoder with RoPE-1D, RMSNorm, SwiGLU FF."""
    dim_head = dim // heads
    return TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=True, dropout=dropout
        ),
        cross_attn=CrossAttentionConfig(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class TransformerDecoder(nn.Module):
    """Opinionated decoder preset: causal self-attn → cross-attn → feed-forward per layer."""

    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self_pos = build_pos_encoding(config.pos_encoding)
        self._decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attn=SelfAttention(config=config.self_attn, pos_encoding=self_pos),
                    cross_attn=CrossAttention(
                        config=config.cross_attn,
                        pos_encoding=NoPosEncoding(NoPosEncodingConfig()),
                    ),
                    ff=build_ff(config.ff),
                    norm_self=build_norm(config.norm),
                    norm_cross=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._decoder(x_input, ctx_input)
