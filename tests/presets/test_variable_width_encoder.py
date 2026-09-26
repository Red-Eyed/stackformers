"""Tests for the variable-width Transformer encoder preset."""

from __future__ import annotations

import inspect
from typing import TypeAlias

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from typing_extensions import TypedDict

from stackformers import (
    GEGLU,
    GELUFFN,
    DistanceBiasConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLU,
    HardSwishGLUConfig,
    LayerNormConfig,
    RelativeDistanceBias,
    ReluSquaredConfig,
    ReluSquaredFF,
    RMSNormConfig,
    RoPE1DConfig,
    RoPE2DConfig,
    RotaryEmbedding2D,
    SelfAttention,
    SelfAttentionConfig,
    SwiGLU,
    SwiGLUConfig,
    VariableWidthEncoderLayerConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    variable_width_encoder_config,
)
from stackformers.layers import (
    NormPlacement,
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.sequence import (
    PackedInput,
    PaddedInput,
    make_packed_input,
    make_padded_input,
    padded_to_packed,
)
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run

B, N, D_IN, D_OUT = 2, 8, 192, 384
NT = 10
D_MODELS = [D_IN, D_IN, D_OUT]
DIM_HEADS = [64, 64, 64]


class LayerSettings(TypedDict, total=False, closed=True):
    """Describe optional per-layer schedules used by factory contract tests."""

    heads: int | list[int]
    causal: bool | list[bool]
    ff_mult: float | list[float]
    dropout: float | list[float]
    norm_placement: NormPlacement | list[NormPlacement]


FeedForwardConfigType: TypeAlias = (
    type[SwiGLUConfig]
    | type[HardSwishGLUConfig]
    | type[GEGLUConfig]
    | type[GELUConfig]
    | type[ReluSquaredConfig]
)

FEED_FORWARD_VARIANTS = [
    pytest.param(SwiGLUConfig, SwiGLU, id="swiglu"),
    pytest.param(HardSwishGLUConfig, HardSwishGLU, id="hardswish-glu"),
    pytest.param(GEGLUConfig, GEGLU, id="geglu"),
    pytest.param(GELUConfig, GELUFFN, id="gelu"),
    pytest.param(ReluSquaredConfig, ReluSquaredFF, id="relu-squared"),
]


def _config_with_feed_forward(
    config_type: FeedForwardConfigType,
) -> VariableWidthTransformerEncoderConfig:
    """Build a variable-width config using one feed-forward type in every block."""
    default = variable_width_encoder_config(d_models=D_MODELS, dim_heads=DIM_HEADS)
    layers = [
        VariableWidthEncoderLayerConfig(
            attn=layer.attn,
            ff=config_type(dim=layer.attn.dim, mult=1.0),
            norm=layer.norm,
            pos_encoding=layer.pos_encoding,
            attn_bias=layer.attn_bias,
        )
        for layer in default.layers
    ]
    return VariableWidthTransformerEncoderConfig(layers=layers)


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
    return variable_width_encoder_config(d_models=D_MODELS, dim_heads=DIM_HEADS)


def test_padded_output_uses_final_width(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Padded leading dimensions survive while the feature width follows the schedule."""
    device, dtype = device_dtype
    encoder = VariableWidthTransformerEncoder(config=config).to(device=device, dtype=dtype)

    output = encoder(padded_input)

    assert output.shape == (B, N, D_OUT)


def test_packed_output_uses_final_width(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Packed token layout survives while the feature width follows the schedule."""
    device, dtype = device_dtype
    encoder = VariableWidthTransformerEncoder(config=config).to(device=device, dtype=dtype)

    output = encoder(packed_input)

    assert output.shape == (NT, D_OUT)


def test_default_factory_builds_complete_layer_configs(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """The convenience factory keeps every default component explicit and inspectable."""
    first, _, last = config.layers

    assert first.attn.dim == D_IN
    assert first.attn.heads == 3
    assert first.attn.dim_head == 64
    assert isinstance(first.ff, SwiGLUConfig)
    assert isinstance(first.norm, RMSNormConfig)
    assert isinstance(first.pos_encoding, RoPE1DConfig)
    assert last.attn.dim == D_OUT
    assert last.attn.heads == 6


def test_config_factory_requires_keyword_arguments() -> None:
    """The variable-width config factory rejects positional construction."""
    factory_parameters = inspect.signature(variable_width_encoder_config).parameters

    assert factory_parameters["d_models"].kind is inspect.Parameter.KEYWORD_ONLY
    assert factory_parameters["dim_heads"].kind is inspect.Parameter.KEYWORD_ONLY


def test_scalar_settings_match_explicit_schedules() -> None:
    """Broadcasting preserves the complete config produced by repeated list values."""
    shared = variable_width_encoder_config(
        d_models=D_MODELS,
        dim_heads=64,
        causal=True,
        ff_mult=2.0,
        dropout=0.1,
        norm_placement="post",
    )
    scheduled = variable_width_encoder_config(
        d_models=D_MODELS,
        dim_heads=DIM_HEADS,
        causal=[True, True, True],
        ff_mult=[2.0, 2.0, 2.0],
        dropout=[0.1, 0.1, 0.1],
        norm_placement=["post", "post", "post"],
    )

    assert shared.layers == scheduled.layers
    assert all(
        isinstance(wrapper.layer, PostNormTransformerLayer)
        for wrapper in VariableWidthTransformerEncoder(shared).layers
    )


def test_mixed_schedules_build_independent_layers() -> None:
    """Each scheduled setting reaches its layer while scalar settings broadcast."""
    config = variable_width_encoder_config(
        d_models=D_MODELS,
        dim_heads=[32, 64, 128],
        causal=[False, True, True],
        ff_mult=[1.0, 2.0, 3.0],
        dropout=0.1,
        norm_placement=["pre", "post", "sandwich"],
    )
    assert [layer.attn.heads for layer in config.layers] == [6, 3, 3]
    assert [layer.attn.causal for layer in config.layers] == [False, True, True]
    assert [layer.ff.mult for layer in config.layers] == [1.0, 2.0, 3.0]
    assert [layer.attn.dropout for layer in config.layers] == [0.1, 0.1, 0.1]
    assert [layer.ff.dropout for layer in config.layers] == [0.1, 0.1, 0.1]
    restored = VariableWidthTransformerEncoderConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    encoder = VariableWidthTransformerEncoder(restored)
    assert type(encoder.get_submodule("layers.0.layer")) is TransformerLayer
    assert type(encoder.get_submodule("layers.1.layer")) is PostNormTransformerLayer
    assert type(encoder.get_submodule("layers.2.layer")) is SandwichNormTransformerLayer
    x = torch.randn(1, 2, D_IN, requires_grad=True)
    output = encoder(make_padded_input(x, torch.ones(1, 2, dtype=torch.bool)))
    output.square().mean().backward()
    assert output.shape == (1, 2, D_OUT)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dropout_schedule_reaches_attention_and_feed_forward() -> None:
    """One per-layer dropout value configures both branches without mutating inputs."""
    dropouts = [0.0, 0.1, 0.2]
    config = variable_width_encoder_config(d_models=D_MODELS, dim_heads=64, dropout=dropouts)
    assert [layer.attn.dropout for layer in config.layers] == [0.0, 0.1, 0.2]
    assert [layer.ff.dropout for layer in config.layers] == [0.0, 0.1, 0.2]
    assert dropouts == [0.0, 0.1, 0.2]


@pytest.mark.parametrize("heads", [3, [3, 5], [1, 2]], ids=["shared", "expanded", "compressed"])
@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_explicit_heads_allow_independent_projection_widths(
    heads: int | list[int], packed: bool
) -> None:
    """Nondivisible model/head widths run both layouts with a finite training signal."""
    config = variable_width_encoder_config(d_models=[10, 14], dim_heads=4, heads=heads)
    expected_heads = heads if isinstance(heads, list) else [heads, heads]
    assert [layer.attn.heads for layer in config.layers] == expected_heads
    encoder = VariableWidthTransformerEncoder(config)
    for index, count in enumerate(expected_heads):
        projection = encoder.get_submodule(f"layers.{index}.layer.self_attn.to_q")
        assert isinstance(projection, nn.Linear)
        assert projection.out_features == count * 4
    x = torch.linspace(-1.0, 1.0, 20).reshape(1, 2, 10).requires_grad_()
    padded = make_padded_input(x, torch.ones(1, 2, dtype=torch.bool))
    output = encoder(padded_to_packed(padded) if packed else padded)
    assert output.shape == ((2, 14) if packed else (1, 2, 14))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_explicit_inferred_heads_preserve_legacy_calls(packed: bool) -> None:
    """Equivalent explicit heads retain old configs, checkpoints, and outputs."""
    legacy = variable_width_encoder_config(d_models=D_MODELS, dim_heads=DIM_HEADS)
    explicit = variable_width_encoder_config(
        d_models=D_MODELS, dim_heads=DIM_HEADS, heads=[3, 3, 6]
    )
    assert legacy == explicit
    assert [layer.attn.heads for layer in legacy.layers] == [3, 3, 6]
    legacy_model = VariableWidthTransformerEncoder(legacy)
    explicit_model = VariableWidthTransformerEncoder(explicit)
    explicit_model.load_state_dict(legacy_model.state_dict(), strict=True)
    x = torch.linspace(-1.0, 1.0, 2 * D_IN).reshape(1, 2, D_IN)
    padded = make_padded_input(x, torch.ones(1, 2, dtype=torch.bool))
    sequence = padded_to_packed(padded) if packed else padded
    torch.testing.assert_close(explicit_model(sequence), legacy_model(sequence), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [0, -1, [3, 0], [3, -1]])
def test_invalid_explicit_heads_are_rejected(heads: int | list[int]) -> None:
    """Explicit counts retain attention's positive-head admission at the public boundary."""
    with pytest.raises(ValidationError, match="greater than 0"):
        variable_width_encoder_config(d_models=[10, 14], dim_heads=4, heads=heads)


def test_explicit_independent_heads_preserve_torch_export() -> None:
    """Strict export preserves the new nondivisible geometries' eager tensor output."""
    config = variable_width_encoder_config(d_models=[10, 14], dim_heads=4, heads=[3, 5])
    encoder = VariableWidthTransformerEncoder(config).eval()
    x = torch.linspace(-1.0, 1.0, 20).reshape(1, 2, 10)
    sequence = make_padded_input(x, torch.ones(1, 2, dtype=torch.bool))
    exported = torch.export.export(encoder, (sequence,), strict=True).module()
    actual: object = exported(sequence)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, encoder(sequence))


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        ({"heads": []}, "heads"),
        ({"heads": [3]}, "heads"),
        ({"causal": []}, "causal"),
        ({"causal": [True]}, "causal"),
        ({"ff_mult": [2.0, 3.0]}, "ff_mult"),
        ({"dropout": [0.0, 0.1, 0.2, 0.3]}, "dropout"),
        ({"norm_placement": ["pre"]}, "norm_placement"),
    ],
)
def test_setting_schedule_lengths_are_validated(settings: LayerSettings, field: str) -> None:
    """Lists must match the layer count; singleton lists do not broadcast."""
    with pytest.raises(ValueError, match=f"{field} must have equal lengths"):
        variable_width_encoder_config(d_models=D_MODELS, dim_heads=64, **settings)


@pytest.mark.parametrize("settings", [{"ff_mult": [1.0, 0.0, 2.0]}, {"dropout": [0.0, 1.1, 0.0]}])
def test_invalid_scheduled_values_are_rejected(settings: LayerSettings) -> None:
    """Per-layer values retain the component models' range validation."""
    with pytest.raises(ValidationError):
        variable_width_encoder_config(d_models=D_MODELS, dim_heads=64, **settings)


def test_direct_config_rejects_mismatched_placements(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Direct configuration validates placement lengths independently of the factory."""
    with pytest.raises(ValidationError, match="norm_placement must have equal lengths"):
        VariableWidthTransformerEncoderConfig(layers=config.layers, norm_placement=["pre"])


def test_config_round_trip_preserves_component_types(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Serialized configs restore every discriminated layer component."""
    restored = VariableWidthTransformerEncoderConfig.model_validate(config.model_dump())

    assert restored == config
    assert isinstance(restored.layers[-1].ff, SwiGLUConfig)
    assert isinstance(restored.layers[-1].pos_encoding, RoPE1DConfig)


def test_encoder_derives_attention_geometry_from_width_schedules(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Each block derives its concrete attention config from the two parallel arrays."""
    encoder = VariableWidthTransformerEncoder(config=config)
    first_attn = encoder.get_submodule("layers.0.layer.self_attn")
    last_attn = encoder.get_submodule("layers.2.layer.self_attn")

    assert isinstance(first_attn, SelfAttention)
    assert first_attn.config.dim == D_IN
    assert first_attn.config.heads == 3
    assert first_attn.config.dim_head == 64
    assert isinstance(last_attn, SelfAttention)
    assert last_attn.config.dim == D_OUT
    assert last_attn.config.heads == 6


def test_explicit_layer_config_selects_every_component() -> None:
    """Direct public configuration controls attention, FF, norm, position, and bias."""
    layer_config = VariableWidthEncoderLayerConfig(
        attn=SelfAttentionConfig(dim=D_IN, heads=3, dim_head=64, window_size=2, qk_norm=True),
        ff=GEGLUConfig(dim=D_IN, mult=1.0),
        norm=LayerNormConfig(dim=D_IN),
        pos_encoding=RoPE2DConfig(dim_head=64),
        attn_bias=DistanceBiasConfig(heads=3, r_max=4.0),
    )
    encoder = VariableWidthTransformerEncoder(
        config=VariableWidthTransformerEncoderConfig(layers=[layer_config])
    )
    layer = encoder.get_submodule("layers.0.layer")
    self_attn = encoder.get_submodule("layers.0.layer.self_attn")

    assert isinstance(layer, TransformerLayer)
    assert isinstance(layer.ff, GEGLU)
    assert isinstance(self_attn, SelfAttention)
    assert isinstance(self_attn.pos_encoding, RotaryEmbedding2D)
    assert isinstance(self_attn.attn_bias, RelativeDistanceBias)
    assert isinstance(encoder.final_norm, nn.LayerNorm)


def test_projection_exists_only_at_width_change(
    config: VariableWidthTransformerEncoderConfig,
) -> None:
    """Equal-width blocks use identity and a stage boundary owns one learned projection."""
    encoder = VariableWidthTransformerEncoder(config=config)
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
        d_models=D_MODELS,
        dim_heads=DIM_HEADS,
        norm_placement=norm_placement,
    )
    encoder = VariableWidthTransformerEncoder(config=config)

    assert all(
        isinstance(encoder.get_submodule(f"layers.{index}.layer"), layer_type)
        for index in range(len(D_MODELS))
    )


@pytest.mark.parametrize(
    ("d_models", "dim_heads", "message"),
    [
        ([], [], "at least one"),
        ([], [64], "at least one"),
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
        variable_width_encoder_config(d_models=d_models, dim_heads=dim_heads)


@pytest.mark.parametrize(
    ("component", "replacement", "message"),
    [
        ("ff", SwiGLUConfig(dim=D_OUT), "ff.dim"),
        ("norm", LayerNormConfig(dim=D_OUT), "norm.dim"),
        ("pos_encoding", RoPE1DConfig(dim_head=128), "pos_encoding.dim_head"),
        ("attn_bias", DistanceBiasConfig(heads=6, r_max=4.0), "attn_bias.heads"),
    ],
)
def test_layer_config_rejects_mismatched_component_dimensions(
    component: str,
    replacement: SwiGLUConfig | LayerNormConfig | RoPE1DConfig | DistanceBiasConfig,
    message: str,
) -> None:
    """A malformed block fails before incompatible components reach tensor operations."""
    base = variable_width_encoder_config(d_models=[D_IN], dim_heads=[64]).layers[0]
    payload = base.model_dump()
    payload[component] = replacement.model_dump()

    with pytest.raises(ValidationError, match=message):
        VariableWidthEncoderLayerConfig.model_validate(payload)


@pytest.mark.parametrize("ff_mult", [0.0, -1.0])
def test_config_rejects_non_positive_ff_multiplier(ff_mult: float) -> None:
    """Invalid shared FF multipliers fail during config validation."""
    with pytest.raises(ValidationError, match="greater than 0"):
        variable_width_encoder_config(
            d_models=D_MODELS,
            dim_heads=DIM_HEADS,
            ff_mult=ff_mult,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_config_rejects_dropout_outside_probability_range(
    dropout: float,
) -> None:
    """Invalid shared dropout values fail during config validation."""
    with pytest.raises(ValidationError, match="greater than or equal to 0|less than or equal to 1"):
        variable_width_encoder_config(
            d_models=D_MODELS,
            dim_heads=DIM_HEADS,
            dropout=dropout,
        )


def test_width_projection_propagates_gradients(device: torch.device) -> None:
    """The stage projection and original-width input both receive training signal."""
    config = variable_width_encoder_config(d_models=D_MODELS, dim_heads=DIM_HEADS)
    encoder = VariableWidthTransformerEncoder(config=config).to(device)
    x = torch.randn(B, N, D_IN, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    encoder(make_padded_input(x, mask)).square().mean().backward()

    projection = encoder.get_submodule("layers.2.projection")
    assert isinstance(projection, nn.Linear)
    assert x.grad is not None
    assert projection.weight.grad is not None


@pytest.mark.parametrize(("config_type", "module_type"), FEED_FORWARD_VARIANTS)
def test_every_feed_forward_variant_is_configurable(
    config_type: FeedForwardConfigType,
    module_type: type[nn.Module],
) -> None:
    """Every public feed-forward config builds the requested block implementation."""
    encoder = VariableWidthTransformerEncoder(config=_config_with_feed_forward(config_type))

    assert all(
        isinstance(encoder.get_submodule(f"layers.{index}.layer.ff"), module_type)
        for index in range(len(D_MODELS))
    )


@pytest.mark.parametrize(("config_type", "_"), FEED_FORWARD_VARIANTS)
@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_variable_width_encoder_is_export_compatible(
    config_type: FeedForwardConfigType,
    _: type[nn.Module],
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Every feed-forward variant exports with static and dynamic shapes."""
    config = _config_with_feed_forward(config_type)
    encoder = VariableWidthTransformerEncoder(config=config)
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
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(encoder, (input,)),
        runtime_args=(resized_input,),
    )
