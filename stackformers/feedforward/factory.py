from __future__ import annotations

from stackformers.feedforward.config import (
    FeedForwardConfig,
    GEGLUConfig,
    ReluSquaredConfig,
    SwiGLUConfig,
)
from stackformers.feedforward.geglu import GEGLU
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.relu_squared import ReluSquaredFF
from stackformers.feedforward.swiglu import SwiGLU


def build_ff(config: FeedForwardConfig) -> FeedForward:
    match config:
        case SwiGLUConfig():
            return SwiGLU(config)
        case GEGLUConfig():
            return GEGLU(config)
        case ReluSquaredConfig():
            return ReluSquaredFF(config)
        case _:
            raise AssertionError(f"Unhandled feedforward config: {type(config)}")
