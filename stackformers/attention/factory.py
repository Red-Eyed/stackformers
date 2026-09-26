from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import (
    AttnBiasConfig,
    DistanceBiasConfig,
    NoAttnBiasConfig,
)
from stackformers.attention.distance_bias import RelativeDistanceBias

if TYPE_CHECKING:
    from stackformers.attention.protocols import AttnBias


def build_attn_bias(config: AttnBiasConfig) -> AttnBias:
    match config:
        case NoAttnBiasConfig():
            return NoAttnBias()
        case DistanceBiasConfig():
            return RelativeDistanceBias(config)
        case _:
            assert_never(config)
