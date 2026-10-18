"""Construct feed-forward modules from discriminated configuration values."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from stackformers.feedforward.config import (
    FeedForwardConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLUConfig,
    PackedGEGLUConfig,
    PackedHardSwishGLUConfig,
    PackedSwiGLUConfig,
    ReluSquaredConfig,
    SwiGLUConfig,
)
from stackformers.feedforward.geglu import GEGLU
from stackformers.feedforward.gelu import GELUFFN
from stackformers.feedforward.hardswish_glu import HardSwishGLU
from stackformers.feedforward.packed_geglu import PackedGEGLU
from stackformers.feedforward.packed_hardswish_glu import PackedHardSwishGLU
from stackformers.feedforward.packed_swiglu import PackedSwiGLU
from stackformers.feedforward.relu_squared import ReluSquaredFF
from stackformers.feedforward.swiglu import SwiGLU

if TYPE_CHECKING:
    from stackformers.feedforward.protocols import FeedForward


def build_ff(config: FeedForwardConfig) -> FeedForward:
    """Build the feed-forward implementation selected by ``config.kind``."""
    match config:
        case PackedGEGLUConfig():
            return PackedGEGLU(config)
        case PackedHardSwishGLUConfig():
            return PackedHardSwishGLU(config)
        case PackedSwiGLUConfig():
            return PackedSwiGLU(config)
        case SwiGLUConfig():
            return SwiGLU(config)
        case HardSwishGLUConfig():
            return HardSwishGLU(config)
        case GEGLUConfig():
            return GEGLU(config)
        case GELUConfig():
            return GELUFFN(config)
        case ReluSquaredConfig():
            return ReluSquaredFF(config)
        case _:
            assert_never(config)
