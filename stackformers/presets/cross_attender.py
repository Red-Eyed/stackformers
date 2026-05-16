from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import SequenceInput


class CrossAttenderConfig(BaseModel):
    attn: CrossAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig = NoPosEncodingConfig()
    num_layers: int = Field(gt=0)


def plain_cross_attender_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> CrossAttenderConfig:
    """Global SDPA cross-attender with RMSNorm and SwiGLU FF, no positional encoding.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return CrossAttenderConfig(
        attn=CrossAttentionConfig(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        num_layers=num_layers,
    )


class CrossAttender(nn.Module):
    """Opinionated cross-attender: queries from x attend to context, no self-attention.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """

    def __init__(self, config: CrossAttenderConfig) -> None:
        super().__init__()
        self.config = config
        pos = build_pos_encoding(config.pos_encoding)
        self._stack = CrossAttenderStack(
            layers=[
                CrossAttenderLayer(
                    cross_attn=CrossAttention(config=config.attn, pos_encoding=pos),
                    ff=build_ff(config.ff),
                    norm_cross=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._stack(x_input, ctx_input)
