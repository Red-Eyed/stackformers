from __future__ import annotations

from stackformers.norm.config import LayerNormConfig, NormConfig, RMSNormConfig
from stackformers.norm.layer_norm import LayerNorm
from stackformers.norm.protocols import Norm
from stackformers.norm.rms import RMSNorm

__all__ = ["NormConfig", "build_norm"]


def build_norm(config: NormConfig) -> Norm:
    match config:
        case RMSNormConfig():
            return RMSNorm(config)
        case LayerNormConfig():
            return LayerNorm(config)
        case _:
            raise AssertionError(f"Unhandled norm config: {type(config)}")
