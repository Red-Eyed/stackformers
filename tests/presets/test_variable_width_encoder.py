"""Tests for the variable-width Transformer encoder preset."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from stackformers.attention.self_attn import SelfAttention
from stackformers.layers import (
    NormPlacement,
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.presets.variable_width_encoder import (
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    variable_width_encoder_config,
)
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input
from tests.export_utils import ExportShapeMode, export_and_run

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


def test_config_stores_architecture_as_structure_of_arrays(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Per-block dimensions are parallel arrays while shared settings remain scalar."""
    assert config.d_models == D_MODELS
    assert config.dim_heads == DIM_HEADS
    assert config.causal is False
    assert config.ff_mult == 4.0
    assert config.dropout == 0.0


def test_config_round_trip_preserves_architecture_schedules(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Serialized configs restore the SoA schedules and shared settings."""
    restored = VariableWidthTransformerEncoderConfig.model_validate(config.model_dump())

    assert restored == config


def test_encoder_derives_attention_geometry_from_width_schedules(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Each block derives its concrete attention config from the two parallel arrays."""
    encoder = VariableWidthTransformerEncoder(config)
    first_attn = encoder.get_submodule("layers.0.layer.self_attn")
    last_attn = encoder.get_submodule("layers.2.layer.self_attn")

    assert isinstance(first_attn, SelfAttention)
    assert first_attn.config.dim == D_IN
    assert first_attn.config.heads == 3
    assert first_attn.config.dim_head == 64
    assert isinstance(last_attn, SelfAttention)
    assert last_attn.config.dim == D_OUT
    assert last_attn.config.heads == 6


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


@pytest.mark.parametrize("ff_mult", [0.0, -1.0])
def test_config_rejects_non_positive_ff_multiplier(ff_mult: float) -> None:
    """Invalid shared FF multipliers fail during config validation."""
    with pytest.raises(ValidationError, match="greater than 0"):
        variable_width_encoder_config(D_MODELS, DIM_HEADS, ff_mult=ff_mult)


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_config_rejects_dropout_outside_probability_range(
    dropout: float,
) -> None:
    """Invalid shared dropout values fail during config validation."""
    with pytest.raises(ValidationError, match="greater than or equal to 0|less than or equal to 1"):
        variable_width_encoder_config(D_MODELS, DIM_HEADS, dropout=dropout)


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


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
def test_variable_width_encoder_is_export_compatible(shape_mode: ExportShapeMode) -> None:
    """Width transitions export with static and dynamic shapes through each backend."""
    config = variable_width_encoder_config(D_MODELS, DIM_HEADS)
    encoder = VariableWidthTransformerEncoder(config)
    x = torch.randn(B, N, D_IN)
    mask = torch.ones(B, N, dtype=torch.bool)
    input = make_padded_input(x, mask)
    resized_input = make_padded_input(
        torch.randn(1, N // 2, D_IN),
        torch.ones(1, N // 2, dtype=torch.bool),
    )
    batch = torch.export.Dim("batch", min=1, max=B)
    tokens = torch.export.Dim("tokens", min=1, max=N)
    shapes = torch.export.ShapesCollection()
    shapes[input.x] = {0: batch, 1: tokens}
    shapes[input.mask] = {0: batch, 1: tokens}
    shapes[input.abs_positions] = {0: batch, 1: tokens}

    export_and_run(
        encoder,
        (input,),
        shape_mode,
        dynamic_shapes=shapes.dynamic_shapes(encoder, (input,)),
        runtime_args=(resized_input,),
    )
