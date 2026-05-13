from __future__ import annotations

import torch.nn as nn

from stackformers.norm.config import LayerNormConfig, NormConfig, RMSNormConfig
from stackformers.norm.protocols import Norm
from stackformers.norm.rms import RMSNorm

__all__ = ["NormConfig", "build_norm"]


def build_norm(config: NormConfig) -> Norm:
    match config:
        case RMSNormConfig():
            return RMSNorm(config)
        case LayerNormConfig():
            return nn.LayerNorm(config.dim, eps=config.eps)
        case _:
            raise AssertionError(f"Unhandled norm config: {type(config)}")
