"""Tests for encoder presets and their public configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from stackformers.layers import (
    NormPlacement,
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.positional.config import (
    LearnedPosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
    RoPENDConfig,
    YaRNConfig,
)
from stackformers.presets.encoder import (
    TransformerEncoder,
    TransformerEncoderConfig,
    node_encoder_config,
    plain_encoder_config,
    windowed_encoder_config,
)
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input
from tests.export_utils import ExportShapeMode, export_and_run

B, N, D, H = 2, 16, 64, 4
NT = 10  # two packed seqs: 6 + 4

EncoderExportVariant: TypeAlias = Literal[
    "plain",
    "yarn",
    "rope2d",
    "rope_nd",
    "learned",
    "windowed",
    "node",
]


@dataclass(frozen=True)
class EncoderExportCase:
    """Hold one encoder and its example and resized padded inputs."""

    model: nn.Module
    example: PaddedInput
    resized: PaddedInput


def _positioned_input(batch: int, tokens: int, dim: int, coords: int) -> PaddedInput:
    """Build an export input with the requested number of coordinate channels."""
    x = torch.randn(batch, tokens, dim)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    positions = torch.rand(batch, tokens, coords)
    return PaddedInput(x=x, mask=mask, abs_positions=positions)


def _encoder_export_case(variant: EncoderExportVariant) -> EncoderExportCase:
    """Build one distinct public encoder configuration and two valid input shapes."""
    dim = 192 if variant == "rope_nd" else D
    example = make_padded_input(torch.randn(2, 6, dim), torch.ones(2, 6, dtype=torch.bool))
    resized = make_padded_input(torch.randn(1, 3, dim), torch.ones(1, 3, dtype=torch.bool))
    ff_mult = 1.0 if variant == "rope_nd" else 1.5
    config = plain_encoder_config(dim, heads=1, num_layers=1, ff_mult=ff_mult)

    if variant == "yarn":
        pos_encoding = RoPE1DConfig(
            dim_head=dim,
            yarn=YaRNConfig(scale=4.0, original_max_seq_len=512),
        )
        config = config.model_copy(update={"pos_encoding": pos_encoding})
    elif variant == "rope2d":
        config = config.model_copy(update={"pos_encoding": RoPE2DConfig(dim_head=dim)})
        example = _positioned_input(2, 6, dim, coords=2)
        resized = _positioned_input(1, 3, dim, coords=2)
    elif variant == "rope_nd":
        pos_encoding = RoPENDConfig(
            dim_head=dim,
            coords=3,
            r_min=0.5,
            r_max=100.0,
        )
        config = config.model_copy(update={"pos_encoding": pos_encoding})
        example = _positioned_input(2, 6, dim, coords=3)
        resized = _positioned_input(1, 3, dim, coords=3)
    elif variant == "learned":
        config = config.model_copy(
            update={"pos_encoding": LearnedPosEncodingConfig(dim_head=dim, max_seq_len=8)}
        )
    elif variant == "windowed":
        config = windowed_encoder_config(
            dim,
            heads=1,
            num_layers=1,
            window_size=2,
            ff_mult=1.5,
        )
    elif variant == "node":
        config = node_encoder_config(
            dim,
            heads=1,
            num_layers=1,
            r_max=4.0,
            ff_mult=1.5,
        )
        example = _positioned_input(2, 6, dim, coords=2)
        resized = _positioned_input(1, 3, dim, coords=2)

    return EncoderExportCase(TransformerEncoder(config), example, resized)


@pytest.fixture
def padded_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def packed_input(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    if not device.type == "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


def test_plain_encoder_padded_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_plain_encoder_packed_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(packed_input)
    assert out.shape == (NT, D)


def test_windowed_encoder_padded_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = windowed_encoder_config(D, H, num_layers=2, window_size=4)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_windowed_encoder_packed_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    cfg = windowed_encoder_config(D, H, num_layers=2, window_size=4)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(packed_input)
    assert out.shape == (NT, D)


def test_plain_encoder_causal(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2, causal=True)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_legacy_encoder_config_defaults_to_pre_norm() -> None:
    """A serialized config from before norm placement existed keeps pre-norm behavior."""
    legacy_payload = plain_encoder_config(D, H, num_layers=2).model_dump()
    legacy_payload.pop("norm_placement")

    restored = TransformerEncoderConfig.model_validate(legacy_payload)

    assert restored.norm_placement == "pre"


@pytest.mark.parametrize(
    ("norm_placement", "layer_type"),
    [
        ("pre", TransformerLayer),
        ("post", PostNormTransformerLayer),
        ("sandwich", SandwichNormTransformerLayer),
        ("reordered", ReorderedNormTransformerLayer),
    ],
)
def test_encoder_builds_selected_norm_topology(
    norm_placement: NormPlacement,
    layer_type: type[TransformerLayerBase],
) -> None:
    """Every supported config value selects its focused layer implementation."""
    config = plain_encoder_config(D, H, num_layers=2, norm_placement=norm_placement)

    encoder = TransformerEncoder(config)

    assert all(isinstance(layer, layer_type) for layer in encoder._encoder.layers)


def test_encoder_config_rejects_unknown_norm_placement() -> None:
    """The Literal-backed Pydantic field rejects unsupported future spellings."""
    payload = plain_encoder_config(D, H, num_layers=2).model_dump()
    payload["norm_placement"] = "future"

    with pytest.raises(ValidationError):
        TransformerEncoderConfig.model_validate(payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed attention requires CUDA")
def test_padded_and_packed_share_weights() -> None:
    """Same model weights handle both padded (inference) and packed (training) inputs."""
    cfg = plain_encoder_config(D, H, num_layers=2)
    enc = TransformerEncoder(cfg).to(device="cuda", dtype=torch.float16)
    padded = make_padded_input(
        torch.randn(B, N, D, device="cuda", dtype=torch.float16),
        torch.ones(B, N, dtype=torch.bool, device="cuda"),
    )
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device="cuda")
    packed = make_packed_input(
        torch.randn(NT, D, device="cuda", dtype=torch.float16), cu, max_seqlen=6
    )
    with torch.no_grad():
        padded_out = enc(padded)
        packed_out = enc(packed)
    assert padded_out.shape == (B, N, D)
    assert packed_out.shape == (NT, D)


@pytest.mark.parametrize(
    "variant",
    ["plain", "yarn", "rope2d", "rope_nd", "learned", "windowed", "node"],
)
@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
def test_encoder_variant_is_export_compatible(
    variant: EncoderExportVariant,
    shape_mode: ExportShapeMode,
) -> None:
    """Every encoder path exports with both static and dynamic input shapes."""
    case = _encoder_export_case(variant)
    batch = torch.export.Dim("batch", min=1, max=2)
    tokens = torch.export.Dim("tokens", min=1, max=8)
    shapes = torch.export.ShapesCollection()
    for tensor in case.example:
        shapes[tensor] = {0: batch, 1: tokens}

    export_and_run(
        case.model,
        (case.example,),
        shape_mode,
        dynamic_shapes=shapes.dynamic_shapes(case.model, (case.example,)),
        runtime_args=(case.resized,),
    )
