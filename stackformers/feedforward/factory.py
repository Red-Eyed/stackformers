from __future__ import annotations

from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.swiglu import SwiGLU


def build_ff(config: FeedForwardConfig) -> FeedForward:
    return SwiGLU(config)
