"""Tests for the package-level public component API."""

from __future__ import annotations

import pytest

import stackformers
import stackformers.presets as presets

PUBLIC_COMPONENT_NAMES = [
    "AttnBiasConfig",
    "DistanceBiasConfig",
    "NoAttnBiasConfig",
    "FeedForwardConfig",
    "SwiGLUConfig",
    "HardSwishGLUConfig",
    "GEGLUConfig",
    "GELUConfig",
    "ReluSquaredConfig",
    "SwiGLU",
    "HardSwishGLU",
    "GEGLU",
    "GELUFFN",
    "ReluSquaredFF",
    "LearnedPosEncodingConfig",
    "RoPE2DConfig",
    "RoPENDConfig",
    "YaRNConfig",
    "LearnedPosEncoding",
    "RotaryEmbeddingND",
    "VariableWidthEncoderLayerConfig",
    "node_encoder_config",
]


@pytest.mark.parametrize("name", PUBLIC_COMPONENT_NAMES)
def test_top_level_exports_public_component(name: str) -> None:
    """Every configurable component is importable from the package root."""
    assert name in stackformers.__all__
    assert hasattr(stackformers, name)


@pytest.mark.parametrize("name", ["VariableWidthEncoderLayerConfig", "node_encoder_config"])
def test_preset_package_exports_public_component(name: str) -> None:
    """Preset-specific configuration entry points remain available from the subpackage."""
    assert name in presets.__all__
    assert hasattr(presets, name)
