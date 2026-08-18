"""Tests for the variable-width Transformer encoder preset."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from stackformers.feedforward.config import SwiGLUConfig
from stackformers.layers import (
    NormPlacement,
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.norm.config import RMSNormConfig
from stackformers.positional.config import RoPE1DConfig
from stackformers.presets.variable_width_encoder import (
    VariableWidthEncoderLayerConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    variable_width_encoder_config,
)
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input

B, N, D_IN, D_OUT = 2, 8, 192, 384
NT = 10
D_MODELS = [D_IN, D_IN, D_OUT]
DIM_HEADS = [64, 64, 64]


@pytest.fixture
def padded_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    """Build a padded sequence whose width matches the preset's first block."""
    device, dtype = device_dtype
    x = torch.randn(B, N, D_IN, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def packed_input(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    """Build a packed sequence on devices supported by variable-length attention."""
    device, dtype = device_dtype
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D_IN, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


@pytest.fixture
def config() -> VariableWidthTransformerEncoderConfig:
    """Build the representative two-stage, three-block configuration."""
    return variable_width_encoder_config(D_MODELS, DIM_HEADS)


def test_padded_output_uses_final_width(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Padded leading dimensions survive while the feature width follows the schedule."""
    device, dtype = device_dtype
    encoder = VariableWidthTransformerEncoder(config).to(device=device, dtype=dtype)

    output = encoder(padded_input)

    assert output.shape == (B, N, D_OUT)


def test_packed_output_uses_final_width(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Packed token layout survives while the feature width follows the schedule."""
    device, dtype = device_dtype
    encoder = VariableWidthTransformerEncoder(config).to(device=device, dtype=dtype)

    output = encoder(packed_input)

    assert output.shape == (NT, D_OUT)


def test_config_expands_explicit_layer_components(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """The convenience schedule remains inspectable as ordinary component configs."""
    first, _, last = config.layers

    assert first.attn.dim == D_IN
    assert first.attn.heads == 3
    assert first.attn.dim_head == 64
    assert isinstance(first.ff, SwiGLUConfig)
    assert first.ff.dim == D_IN
    assert isinstance(first.norm, RMSNormConfig)
    assert first.norm.dim == D_IN
    assert isinstance(first.pos_encoding, RoPE1DConfig)
    assert first.pos_encoding.dim_head == 64
    assert last.attn.dim == D_OUT
    assert last.attn.heads == 6


def test_config_round_trip_preserves_component_types(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Serialized schedules restore every discriminated collaborator configuration."""
    restored = VariableWidthTransformerEncoderConfig.model_validate(config.model_dump())

    assert restored == config
    assert isinstance(restored.layers[-1].ff, SwiGLUConfig)
    assert isinstance(restored.layers[-1].pos_encoding, RoPE1DConfig)


def test_projection_exists_only_at_width_change(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Equal-width blocks use identity and a stage boundary owns one learned projection."""
    encoder = VariableWidthTransformerEncoder(config)
    first_projection = encoder.get_submodule("layers.0.projection")
    second_projection = encoder.get_submodule("layers.1.projection")
    stage_projection = encoder.get_submodule("layers.2.projection")

    assert isinstance(first_projection, nn.Identity)
    assert isinstance(second_projection, nn.Identity)
    assert isinstance(stage_projection, nn.Linear)
    assert stage_projection.in_features == D_IN
    assert stage_projection.out_features == D_OUT
    assert stage_projection.bias is None


@pytest.mark.parametrize(
    ("norm_placement", "layer_type"),
    [
        ("pre", TransformerLayer),
        ("post", PostNormTransformerLayer),
        ("sandwich", SandwichNormTransformerLayer),
        ("reordered", ReorderedNormTransformerLayer),
    ],
)
def test_norm_placement_selects_each_block_topology(
    norm_placement: NormPlacement,
    layer_type: type[TransformerLayerBase],
) -> None:
    """The shared placement setting selects the focused class for every block."""
    config = variable_width_encoder_config(
        D_MODELS,
        DIM_HEADS,
        norm_placement=norm_placement,
    )
    encoder = VariableWidthTransformerEncoder(config)

    assert all(
        isinstance(encoder.get_submodule(f"layers.{index}.layer"), layer_type)
        for index in range(len(D_MODELS))
    )


@pytest.mark.parametrize(
    ("d_models", "dim_heads", "message"),
    [
        ([], [], "at least one"),
        ([192, 384], [64], "equal lengths"),
        ([192, 250], [64, 64], "must be divisible"),
        ([192, 0], [64, 64], "must be positive"),
    ],
)
def test_invalid_width_schedules_are_rejected(
    d_models: list[int],
    dim_heads: list[int],
    message: str,
) -> None:
    """Malformed schedules fail at construction instead of inside an attention kernel."""
    with pytest.raises(ValueError, match=message):
        variable_width_encoder_config(d_models, dim_heads)


@pytest.mark.parametrize(
    ("component", "replacement", "message"),
    [
        ("ff", SwiGLUConfig(dim=D_OUT), "ff.dim"),
        ("norm", RMSNormConfig(dim=D_OUT), "norm.dim"),
        ("pos_encoding", RoPE1DConfig(dim_head=128), "pos_encoding.dim_head"),
    ],
)
def test_layer_config_rejects_mismatched_component_dimensions(
    component: str,
    replacement: SwiGLUConfig | RMSNormConfig | RoPE1DConfig,
    message: str,
) -> None:
    """Explicit collaborators fail before mismatched dimensions reach tensor operations."""
    base = variable_width_encoder_config([D_IN], [64]).layers[0]
    payload = base.model_dump()
    payload[component] = replacement.model_dump()

    with pytest.raises(ValidationError, match=message):
        VariableWidthEncoderLayerConfig.model_validate(payload)


def test_width_projection_propagates_gradients(device: torch.device) -> None:
    """The stage projection and original-width input both receive training signal."""
    config = variable_width_encoder_config(D_MODELS, DIM_HEADS)
    encoder = VariableWidthTransformerEncoder(config).to(device)
    x = torch.randn(B, N, D_IN, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    encoder(make_padded_input(x, mask)).square().mean().backward()

    projection = encoder.get_submodule("layers.2.projection")
    assert isinstance(projection, nn.Linear)
    assert x.grad is not None
    assert projection.weight.grad is not None


def test_variable_width_encoder_is_torch_export_compatible() -> None:
    """Width transitions remain traceable as tensor-only sequence transformations."""
    config = variable_width_encoder_config(D_MODELS, DIM_HEADS)
    encoder = VariableWidthTransformerEncoder(config)
    x = torch.randn(B, N, D_IN)
    mask = torch.ones(B, N, dtype=torch.bool)
    input = make_padded_input(x, mask)

    exported = torch.export.export(encoder, (input,))

    assert torch.equal(exported.module()(input), encoder(input))
