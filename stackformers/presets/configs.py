from __future__ import annotations

import torch.nn as nn

from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.norm.config import LayerNormConfig, RMSNormConfig
from stackformers.norm.protocols import Norm
from stackformers.norm.rms import RMSNorm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D

NormConfig = RMSNormConfig | LayerNormConfig


def build_norm(config: NormConfig) -> Norm:
    match config:
        case RMSNormConfig():
            return RMSNorm(config)
        case LayerNormConfig():
            return nn.LayerNorm(config.dim, eps=config.eps)
        case _:
            raise AssertionError(f"Unhandled norm config: {type(config)}")


def build_ff(config: FeedForwardConfig) -> FeedForward:
    return SwiGLU(config)


def build_pos_encoding(config: PosEncodingConfig) -> PosEncoding:
    match config:
        case RoPE1DConfig():
            return RotaryEmbedding1D(config)
        case NoPosEncodingConfig():
            return NoPosEncoding(config)
        case _:
            raise AssertionError(f"Unhandled pos encoding config: {type(config)}")
