"""SwiGLU layout selection and discriminated JSON configuration."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from stackformers.feedforward.config import FeedForwardConfig, PackedSwiGLUConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.packed_swiglu import PackedSwiGLU
from stackformers.feedforward.swiglu import SwiGLU


@pytest.fixture(params=["separate", "packed"])
def config(request: pytest.FixtureRequest) -> SwiGLUConfig | PackedSwiGLUConfig:
    """Supply the two concrete configuration payloads at identical widths."""
    match request.param:
        case "separate":
            return SwiGLUConfig(dim=24)
        case "packed":
            return PackedSwiGLUConfig(dim=24)
        case _:
            pytest.fail("unknown SwiGLU test variant")


def test_swiglu_variant_roundtrip(config: SwiGLUConfig | PackedSwiGLUConfig) -> None:
    """The feed-forward enum dispatches from the saved concrete config variant."""
    restored = TypeAdapter(FeedForwardConfig).validate_json(config.model_dump_json())
    assert type(restored) is type(config)
    match config:
        case SwiGLUConfig():
            assert type(build_ff(restored)) is SwiGLU
        case PackedSwiGLUConfig():
            assert type(build_ff(restored)) is PackedSwiGLU


def test_conversion_config_variants() -> None:
    """The destination config describes the checkpoint layout after either transition."""
    packed = SwiGLU(SwiGLUConfig(dim=24)).to_packed()
    assert type(packed.config) is PackedSwiGLUConfig
    assert type(packed.unpack().config) is SwiGLUConfig
