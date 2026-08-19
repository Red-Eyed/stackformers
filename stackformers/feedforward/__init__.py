"""Public feed-forward configurations, implementations, and builder."""

from stackformers.feedforward.config import (
    FeedForwardConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLUConfig,
    ReluSquaredConfig,
    SwiGLUConfig,
)
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.geglu import GEGLU
from stackformers.feedforward.gelu import GELUFFN
from stackformers.feedforward.hardswish_glu import HardSwishGLU
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.relu_squared import ReluSquaredFF
from stackformers.feedforward.swiglu import SwiGLU

__all__ = [
    "FeedForward",
    "FeedForwardConfig",
    "GEGLU",
    "GEGLUConfig",
    "GELUFFN",
    "GELUConfig",
    "HardSwishGLU",
    "HardSwishGLUConfig",
    "ReluSquaredConfig",
    "ReluSquaredFF",
    "SwiGLU",
    "SwiGLUConfig",
    "build_ff",
]
