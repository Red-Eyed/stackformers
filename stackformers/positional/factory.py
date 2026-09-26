from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from stackformers.positional.config import (
    LearnedPosEncodingConfig,
    NoPosEncodingConfig,
    PosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
    RoPENDConfig,
)
from stackformers.positional.learned import LearnedPosEncoding
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.positional.rope_nd import RotaryEmbeddingND

if TYPE_CHECKING:
    from stackformers.positional.protocols import PosEncoding


def build_pos_encoding(config: PosEncodingConfig) -> PosEncoding:
    match config:
        case RoPE1DConfig():
            return RotaryEmbedding1D(config)
        case RoPE2DConfig():
            return RotaryEmbedding2D(config)
        case RoPENDConfig():
            return RotaryEmbeddingND(config)
        case NoPosEncodingConfig():
            return NoPosEncoding(config)
        case LearnedPosEncodingConfig():
            return LearnedPosEncoding(config)
        case _:
            assert_never(config)
