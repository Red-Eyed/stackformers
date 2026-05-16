from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels.config import SDPAKernelConfig, VarlenSDPAKernelConfig
from stackformers.attention.kernels.factory import build_kernel
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import SequenceInput


class CrossAttenderConfig(BaseModel):
    attn: AttentionConfig  # causal is always False; kernel defaults to SDPA
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
    """SDPA cross-attender with RMSNorm and SwiGLU FF, no positional encoding."""
    dim_head = dim // heads
    return CrossAttenderConfig(
        attn=AttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel=SDPAKernelConfig()
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        num_layers=num_layers,
    )


def packed_cross_attender_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> CrossAttenderConfig:
    """Varlen SDPA cross-attender with RMSNorm, SwiGLU FF, and RoPE-1D.

    Pass PackedInput during training; same model accepts PaddedInput at inference.
    """
    dim_head = dim // heads
    return CrossAttenderConfig(
        attn=AttentionConfig(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            kernel=VarlenSDPAKernelConfig(),
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class CrossAttender(nn.Module):
    """Opinionated cross-attender preset: queries from x attend to context, no self-attention.

    Pass PackedInput for training (no padding overhead) and PaddedInput for inference
    — same model, same weights, no class swap required.
    """

    def __init__(self, config: CrossAttenderConfig) -> None:
        super().__init__()
        self.config = config

        cross_attn_cfg = config.attn.model_copy(update={"causal": False})
        pos = build_pos_encoding(config.pos_encoding)

        self._stack = CrossAttenderStack(
            layers=[
                CrossAttenderLayer(
                    cross_attn=CrossAttention(
                        config=cross_attn_cfg,
                        pos_encoding=pos,
                        kernel=build_kernel(cross_attn_cfg),
                    ),
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
