"""Tests for decoder presets and normalization-topology selection."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from stackformers.decoder import (
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.norm.config import NormPlacement
from stackformers.presets.decoder import (
    TransformerDecoder,
    TransformerDecoderConfig,
    plain_decoder_config,
)
from stackformers.sequence import PaddedInput, make_padded_input

B, N, S, D, H = 2, 8, 12, 64, 4  # N=target len, S=context len


@pytest.fixture
def target_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def context_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_plain_decoder_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    target_input: PaddedInput,
    context_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_decoder_config(D, H, num_layers=2)
    out = TransformerDecoder(cfg).to(device=device, dtype=dtype)(target_input, context_input)
    assert out.shape == (B, N, D)


def test_legacy_decoder_config_defaults_to_pre_norm() -> None:
    """A serialized config without placement keeps the original decoder topology."""
    legacy_payload = plain_decoder_config(D, H, num_layers=2).model_dump()
    legacy_payload.pop("norm_placement")

    restored = TransformerDecoderConfig.model_validate(legacy_payload)

    assert restored.norm_placement == "pre"


@pytest.mark.parametrize(
    ("norm_placement", "layer_type"),
    [
        ("pre", DecoderLayer),
        ("post", PostNormDecoderLayer),
        ("sandwich", SandwichNormDecoderLayer),
        ("reordered", ReorderedNormDecoderLayer),
    ],
)
def test_decoder_builds_selected_norm_topology(
    norm_placement: NormPlacement,
    layer_type: type[DecoderLayerBase],
) -> None:
    """Every supported config value selects its focused decoder implementation."""
    config = plain_decoder_config(D, H, num_layers=2, norm_placement=norm_placement)

    decoder = TransformerDecoder(config)

    assert all(isinstance(layer, layer_type) for layer in decoder._decoder.layers)


def test_decoder_config_rejects_unknown_norm_placement() -> None:
    """The shared Literal rejects unsupported decoder topology names."""
    payload = plain_decoder_config(D, H, num_layers=2).model_dump()
    payload["norm_placement"] = "future"

    with pytest.raises(ValidationError):
        TransformerDecoderConfig.model_validate(payload)
