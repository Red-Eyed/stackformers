from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from stackformers.norm.config import LayerNormConfig, NormConfig, RMSNormConfig

if TYPE_CHECKING:
    from stackformers.norm.protocols import Norm

__all__ = ["NormConfig", "build_norm"]


def build_norm(config: NormConfig) -> Norm:
    match config:
        case RMSNormConfig():
            return nn.RMSNorm(config.dim, eps=config.eps)
        case LayerNormConfig():
            return nn.LayerNorm(config.dim, eps=config.eps)
        case _:
            raise AssertionError(f"Unhandled norm config: {type(config)}")
