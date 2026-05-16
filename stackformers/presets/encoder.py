from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.layers import TransformerLayer
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import SequenceInput


class TransformerEncoderConfig(BaseModel):
    attn: SelfAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    num_layers: int = Field(gt=0)


def plain_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Global SDPA encoder with RoPE-1D, RMSNorm, and SwiGLU FF.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=SelfAttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


def windowed_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    window_size: int,
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Sliding-window encoder — O(n·w) attention for long sequences.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=SelfAttentionConfig(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            dropout=dropout,
            window_size=window_size,
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class TransformerEncoder(nn.Module):
    """Opinionated encoder preset. Pass PaddedInput for inference, PackedInput for training."""

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        pos = build_pos_encoding(config.pos_encoding)
        self._encoder = Encoder(
            layers=[
                TransformerLayer(
                    self_attn=SelfAttention(config=config.attn, pos_encoding=pos),
                    ff=build_ff(config.ff),
                    norm_attn=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, input: SequenceInput) -> Tensor:
        return self._encoder(input)
