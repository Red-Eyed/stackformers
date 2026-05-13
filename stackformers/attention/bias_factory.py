from __future__ import annotations

from stackformers.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.attention.bias_config import ALiBiConfig, BiasBuilderConfig, NoBiasConfig
from stackformers.attention.protocols import AttnBiasBuilder


def build_bias_builder(config: BiasBuilderConfig, heads: int, causal: bool) -> AttnBiasBuilder:
    match config:
        case NoBiasConfig():
            return NoBiasBuilder()
        case ALiBiConfig():
            return ALiBiBuilder(heads=heads, causal=causal)
        case _:
            raise AssertionError(f"Unhandled bias builder config: {type(config)}")
