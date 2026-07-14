from __future__ import annotations

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import (
    AttnBiasConfig,
    DistanceBiasConfig,
    NoAttnBiasConfig,
)
from stackformers.attention.distance_bias import RelativeDistanceBias
from stackformers.attention.protocols import AttnBias


def build_attn_bias(config: AttnBiasConfig) -> AttnBias:
    match config:
        case NoAttnBiasConfig():
            return NoAttnBias()
        case DistanceBiasConfig():
            return RelativeDistanceBias(config)
        case _:
            raise AssertionError(f"Unhandled attn bias config: {type(config)}")
