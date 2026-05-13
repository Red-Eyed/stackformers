from __future__ import annotations

from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D


def build_pos_encoding(config: PosEncodingConfig) -> PosEncoding:
    match config:
        case RoPE1DConfig():
            return RotaryEmbedding1D(config)
        case NoPosEncodingConfig():
            return NoPosEncoding(config)
        case _:
            raise AssertionError(f"Unhandled pos encoding config: {type(config)}")
