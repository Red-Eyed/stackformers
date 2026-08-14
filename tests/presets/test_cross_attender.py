"""Tests for cross-attender presets and normalization-topology selection."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from stackformers.cross_attender import (
    CrossAttenderLayer,
    CrossAttenderLayerBase,
    PostNormCrossAttenderLayer,
    ReorderedNormCrossAttenderLayer,
    SandwichNormCrossAttenderLayer,
)
from stackformers.norm.config import NormPlacement
from stackformers.presets.cross_attender import (
    CrossAttender,
    CrossAttenderConfig,
    plain_cross_attender_config,
)
from stackformers.sequence import PaddedInput, make_padded_input

B, Nq, S, D, H = 2, 8, 12, 64, 4  # Nq=query len, S=context len


@pytest.fixture
def query_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, Nq, D, device=device, dtype=dtype)
    mask = torch.ones(B, Nq, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def context_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_plain_cross_attender_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    query_input: PaddedInput,
    context_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_cross_attender_config(D, H, num_layers=2)
    out = CrossAttender(cfg).to(device=device, dtype=dtype)(query_input, context_input)
    assert out.shape == (B, Nq, D)


def test_legacy_cross_attender_config_defaults_to_pre_norm() -> None:
    """A serialized config without placement keeps the original cross-attender topology."""
    legacy_payload = plain_cross_attender_config(D, H, num_layers=2).model_dump()
    legacy_payload.pop("norm_placement")

    restored = CrossAttenderConfig.model_validate(legacy_payload)

    assert restored.norm_placement == "pre"


@pytest.mark.parametrize(
    ("norm_placement", "layer_type"),
    [
        ("pre", CrossAttenderLayer),
        ("post", PostNormCrossAttenderLayer),
        ("sandwich", SandwichNormCrossAttenderLayer),
        ("reordered", ReorderedNormCrossAttenderLayer),
    ],
)
def test_cross_attender_builds_selected_norm_topology(
    norm_placement: NormPlacement,
    layer_type: type[CrossAttenderLayerBase],
) -> None:
    """Every supported config value selects its focused cross-attender implementation."""
    config = plain_cross_attender_config(
        D,
        H,
        num_layers=2,
        norm_placement=norm_placement,
    )

    cross_attender = CrossAttender(config)

    assert all(isinstance(layer, layer_type) for layer in cross_attender._stack.layers)


def test_cross_attender_config_rejects_unknown_norm_placement() -> None:
    """The shared Literal rejects unsupported cross-attender topology names."""
    payload = plain_cross_attender_config(D, H, num_layers=2).model_dump()
    payload["norm_placement"] = "future"

    with pytest.raises(ValidationError):
        CrossAttenderConfig.model_validate(payload)
