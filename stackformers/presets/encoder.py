from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias_config import BiasBuilderConfig, NoBiasConfig
from stackformers.attention.bias_factory import build_bias_builder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels.config import (
    SDPAKernelConfig,
    VarlenSDPAKernelConfig,
    WindowedSDPAKernelConfig,
)
from stackformers.attention.kernels.factory import build_kernel
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.layers import TransformerLayer
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import PackedInput, SequenceInput


class TransformerEncoderConfig(BaseModel):
    attn: AttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    bias: BiasBuilderConfig = NoBiasConfig()
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
    """Padded full-sequence SDPA encoder with RoPE-1D, RMSNorm, and SwiGLU FF.

    The default choice for most tasks. Accepts PaddedInput; padding is handled
    via an additive attention mask so no sequence packing is required.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=AttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout,
            kernel=SDPAKernelConfig(),
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        bias=NoBiasConfig(),
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
    """Padded local-window SDPA encoder — O(n·w) attention for long sequences.

    Each token attends only within a sliding window of size `window_size`.
    Accepts PaddedInput.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=AttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout,
            kernel=WindowedSDPAKernelConfig(window_size=window_size),
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        bias=NoBiasConfig(),
        num_layers=num_layers,
    )


def packed_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Packed varlen SDPA encoder — no padding overhead, requires PackedInput.

    Sequences of different lengths are concatenated into a single flat tensor.
    Best for batches with high length variance.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=AttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout,
            kernel=VarlenSDPAKernelConfig(),
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        bias=NoBiasConfig(),
        num_layers=num_layers,
    )


class TransformerEncoder(nn.Module):
    """Opinionated encoder preset: norm, ff, pos-encoding, kernel, and bias builder from config."""

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config

        pos = build_pos_encoding(config.pos_encoding)

        self._encoder = Encoder(
            layers=[
                TransformerLayer(
                    self_attn=SelfAttention(
                        config=config.attn,
                        pos_encoding=pos,
                        bias_builder=build_bias_builder(config.bias, config.attn.heads),
                        kernel=build_kernel(config.attn),
                    ),
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


class PackedTransformerEncoder(nn.Module):
    """Encoder preset for packed (variable-length) sequences.

    Uses PackedSelfAttention — no padding compute or memory overhead.
    Accepts PackedInput only.

    State dict is identical to TransformerEncoder built from the same config,
    so you can train packed and export padded::

        cfg = packed_encoder_config(dim=512, heads=8, num_layers=6)
        train_model = PackedTransformerEncoder(cfg)   # trains on PackedInput
        # ... training loop ...
        export_model = TransformerEncoder(cfg)        # accepts PaddedInput
        export_model.load_state_dict(train_model.state_dict())
    """

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config

        pos = build_pos_encoding(config.pos_encoding)

        self._encoder = Encoder(
            layers=[
                TransformerLayer(
                    self_attn=PackedSelfAttention(
                        config=config.attn,
                        pos_encoding=pos,
                        kernel=build_kernel(config.attn),
                    ),
                    ff=build_ff(config.ff),
                    norm_attn=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, input: PackedInput) -> Tensor:
        return self._encoder(input)
