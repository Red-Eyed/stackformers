from __future__ import annotations

from stackformers.positional.config import (
    LearnedPosEncodingConfig,
    NoPosEncodingConfig,
    PosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
)
from stackformers.positional.learned import LearnedPosEncoding
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D


def build_pos_encoding(config: PosEncodingConfig) -> PosEncoding:
    match config:
        case RoPE1DConfig():
            return RotaryEmbedding1D(config)
        case RoPE2DConfig():
            return RotaryEmbedding2D(config)
        case NoPosEncodingConfig():
            return NoPosEncoding(config)
        case LearnedPosEncodingConfig():
            return LearnedPosEncoding(config)
        case _:
            raise AssertionError(f"Unhandled pos encoding config: {type(config)}")
