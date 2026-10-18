"""Recursive conversion preserves model values, shared children, and checkpoint metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple, assert_never

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers import (
    GEGLU,
    CachedCrossAttentionWrapper,
    CachedSelfAttentionWrapper,
    CrossAttender,
    CrossAttenderConfig,
    CrossAttention,
    CrossAttentionConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLU,
    HardSwishGLUConfig,
    PackedCrossAttention,
    PackedSelfAttention,
    SelfAttention,
    SelfAttentionConfig,
    SwiGLU,
    SwiGLUConfig,
    TransformerDecoder,
    TransformerDecoderConfig,
    TransformerEncoder,
    TransformerEncoderConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    plain_cross_attender_config,
    plain_decoder_config,
    plain_encoder_config,
    to_packed,
    to_unpacked,
    variable_width_encoder_config,
)
from stackformers.positional.config import LearnedPosEncodingConfig
from stackformers.positional.learned import LearnedPosEncoding
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import (
    PackedInput,
    PaddedInput,
    make_padded_input,
    padded_to_packed,
)
from tests.conftest import atol

if TYPE_CHECKING:
    from pathlib import Path

    from stackformers.norm.config import NormPlacement


Layer = SelfAttention | CrossAttention | SwiGLU | GEGLU | HardSwishGLU
Preset = TransformerEncoder | TransformerDecoder | CrossAttender | VariableWidthTransformerEncoder


@pytest.fixture(params=["self", "cross", "swiglu", "geglu", "hardswish"])
def layer(request: pytest.FixtureRequest, device_dtype: tuple[torch.device, torch.dtype]) -> Layer:
    """Build every supported layer with a frozen group and distinct child training mode."""
    match request.param:
        case "self":
            module = SelfAttention(
                SelfAttentionConfig(dim=24, heads=3, dim_head=8, kv_heads=1, qk_norm=True),
                LearnedPosEncoding(LearnedPosEncodingConfig(dim_head=8, max_seq_len=16)),
            )
            module.to_q.requires_grad_(False)
            module.to_k.requires_grad_(False)
            module.to_v.requires_grad_(False)
        case "cross":
            module = CrossAttention(
                CrossAttentionConfig(dim=24, heads=3, dim_head=8, kv_heads=1, qk_norm=True),
                LearnedPosEncoding(LearnedPosEncodingConfig(dim_head=8, max_seq_len=16)),
            )
            module.to_k.requires_grad_(False)
            module.to_v.requires_grad_(False)
        case "swiglu":
            module = SwiGLU(SwiGLUConfig(dim=24))
            module.w1.requires_grad_(False)
            module.w2.requires_grad_(False)
        case "geglu":
            module = GEGLU(GEGLUConfig(dim=24))
            module.w1.requires_grad_(False)
            module.w2.requires_grad_(False)
        case "hardswish":
            module = HardSwishGLU(HardSwishGLUConfig(dim=24))
            module.w1.requires_grad_(False)
            module.w2.requires_grad_(False)
        case _:
            pytest.fail("unknown layer")
    device, dtype = device_dtype
    module.to(device=device, dtype=dtype).eval()
    module.dropout.train()
    return module


@pytest.fixture(params=["padded", "packed"])
def inputs(
    request: pytest.FixtureRequest, device_dtype: tuple[torch.device, torch.dtype]
) -> tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput]:
    """Use masked batches with different query and context lengths in both layouts."""
    device, dtype = device_dtype
    query = make_padded_input(
        torch.randn(2, 5, 24, device=device, dtype=dtype),
        torch.tensor([[True] * 5, [True] * 3 + [False] * 2], device=device),
    )
    context = make_padded_input(
        torch.randn(2, 7, 24, device=device, dtype=dtype),
        torch.tensor([[True] * 7, [True] * 4 + [False] * 3], device=device),
    )
    match request.param:
        case "padded":
            return query, context
        case "packed":
            return padded_to_packed(query), padded_to_packed(context)
        case _:
            pytest.fail("unknown input layout")


def _run(
    module: nn.Module, inputs: tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput]
) -> Tensor:
    """Route tensors through each layer or preset's existing input contract."""
    query, context = inputs
    match module:
        case (
            SelfAttention()
            | PackedSelfAttention()
            | TransformerEncoder()
            | VariableWidthTransformerEncoder()
        ):
            return module(query)
        case CrossAttention() | PackedCrossAttention() | TransformerDecoder() | CrossAttender():
            match inputs:
                case (PaddedInput(), PaddedInput()) | (PackedInput(), PackedInput()):
                    return module(*inputs)
                case _:
                    assert_never(inputs)
        case _:
            output: Tensor = module(query.x)
            return output


def _assert_state_equal(actual: nn.Module, expected: dict[str, Tensor]) -> None:
    """Check exact checkpoint keys and weight values, independent of output tolerance."""
    state = actual.state_dict()
    assert state.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def test_root_roundtrip(
    layer: Layer, inputs: tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput]
) -> None:
    """The returned root changes type while outputs, flags, and weights survive conversion."""
    expected = _run(layer, inputs)
    state = {name: value.clone() for name, value in layer.state_dict().items()}
    flags = {name: value.requires_grad for name, value in layer.named_parameters()}
    modes = {name: child.training for name, child in layer.named_modules()}
    packed = to_packed(layer)
    assert packed is not layer
    assert to_packed(packed) is packed
    torch.testing.assert_close(
        _run(packed, inputs), expected, atol=atol(expected.dtype), rtol=atol(expected.dtype)
    )
    restored = to_unpacked(packed)
    assert type(restored) is type(layer)
    assert to_unpacked(restored) is restored
    _assert_state_equal(restored, state)
    assert {name: value.requires_grad for name, value in restored.named_parameters()} == flags
    assert {name: child.training for name, child in restored.named_modules()} == modes
    torch.testing.assert_close(
        _run(restored, inputs), expected, atol=atol(expected.dtype), rtol=atol(expected.dtype)
    )


class SharedTree(NamedTuple):
    """Keep typed handles to every aliased registration in the test graph."""

    root: nn.ModuleDict
    chain: nn.Sequential
    aliases: nn.ModuleList
    unrelated: nn.Linear


@pytest.fixture
def shared_tree(layer: Layer) -> SharedTree:
    """Register a layer multiple times, with an external reference to its position encoder."""
    chain = nn.Sequential(layer)
    aliases = nn.ModuleList([layer, layer])
    unrelated = nn.Linear(24, 24)
    root = nn.ModuleDict({"chain": chain, "aliases": aliases, "unrelated": unrelated})
    root.add_module("unused", None)
    match layer:
        case SelfAttention() | CrossAttention():
            assert isinstance(layer.pos_encoding, nn.Module)
            root["position"] = layer.pos_encoding
    return SharedTree(root, chain, aliases, unrelated)


def test_nested_shared_references(shared_tree: SharedTree, layer: Layer) -> None:
    """Every child slot receives the same replacement, including duplicate sibling names."""
    root, chain, aliases, unrelated = shared_tree
    assert to_packed(root) is root
    assert chain[0] is aliases[0] is aliases[1]
    assert chain[0] is not layer
    assert root["unrelated"] is unrelated
    match chain[0]:
        case PackedSelfAttention() | PackedCrossAttention() as packed:
            assert packed.pos_encoding is root["position"]
    assert to_unpacked(root) is root
    assert chain[0] is aliases[0] is aliases[1]
    assert type(chain[0]) is type(layer)


@pytest.fixture
def invalid_tree() -> nn.Sequential:
    """Fail on a later layer after an earlier replacement has already been prepared."""
    first = SwiGLU(SwiGLUConfig(dim=24))
    second = GEGLU(GEGLUConfig(dim=24))
    second.w1.requires_grad_(False)
    return nn.Sequential(first, second)


def test_trainability_failure_leaves_tree_unchanged(invalid_tree: nn.Sequential) -> None:
    """Admission errors do not leave a model partially converted."""
    children = tuple(invalid_tree.children())
    state = {name: value.clone() for name, value in invalid_tree.state_dict().items()}
    with pytest.raises(ValueError, match="requires_grad"):
        to_packed(invalid_tree)
    assert tuple(invalid_tree.children()) == children
    _assert_state_equal(invalid_tree, state)


@pytest.fixture(params=["self", "cross"])
def cached_tree(request: pytest.FixtureRequest) -> nn.Sequential:
    """Place a cache executor after a convertible module to verify rejection before mutation."""
    match request.param:
        case "self":
            wrapper = CachedSelfAttentionWrapper(
                SelfAttention(
                    SelfAttentionConfig(dim=24, heads=3, dim_head=8, causal=True),
                    NoPosEncoding(),
                )
            )
        case "cross":
            wrapper = CachedCrossAttentionWrapper(
                CrossAttention(CrossAttentionConfig(dim=24, heads=3, dim_head=8), NoPosEncoding())
            )
        case _:
            pytest.fail("unknown cache wrapper")
    return nn.Sequential(SwiGLU(SwiGLUConfig(dim=24)), wrapper)


def test_cache_wrapper_rejected(cached_tree: nn.Sequential) -> None:
    """A helper must not replace projections that cache execution accesses by name."""
    children = tuple(cached_tree.children())
    with pytest.raises(TypeError, match="cache wrappers"):
        to_packed(cached_tree)
    assert tuple(cached_tree.children()) == children
    assert to_unpacked(cached_tree) is cached_tree


class CustomSwiGLU(SwiGLU):
    """Represent a user override whose forward behavior conversion must retain."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Deliberately differ from the built-in SwiGLU equation."""
        return super().forward(x) * 2


@pytest.fixture
def custom_layer(device_dtype: tuple[torch.device, torch.dtype]) -> CustomSwiGLU:
    """Build a subclass on the same device/dtype matrix as supported layers."""
    device, dtype = device_dtype
    return CustomSwiGLU(SwiGLUConfig(dim=24)).to(device=device, dtype=dtype).eval()


def test_custom_subclass_preserved(
    custom_layer: CustomSwiGLU,
    inputs: tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput],
) -> None:
    """Matching a base class never silently discards a user's overridden equation."""
    expected = _run(custom_layer, inputs)
    assert to_packed(custom_layer) is custom_layer
    assert to_unpacked(custom_layer) is custom_layer
    torch.testing.assert_close(_run(custom_layer, inputs), expected, rtol=0, atol=0)


@pytest.fixture(params=["pre", "post", "sandwich", "reordered"])
def placement(request: pytest.FixtureRequest) -> NormPlacement:
    """Cover every residual and normalization topology."""
    match request.param:
        case "pre" | "post" | "sandwich" | "reordered" as placement:
            return placement
        case _:
            pytest.fail("unknown normalization placement")


@pytest.fixture(params=["swiglu", "geglu", "hardswish", "gelu"])
def ff_config(
    request: pytest.FixtureRequest,
) -> SwiGLUConfig | GEGLUConfig | HardSwishGLUConfig | GELUConfig:
    """Exercise every convertible gate through a complete preset."""
    match request.param:
        case "swiglu":
            return SwiGLUConfig(dim=24)
        case "geglu":
            return GEGLUConfig(dim=24)
        case "hardswish":
            return HardSwishGLUConfig(dim=24)
        case "gelu":
            return GELUConfig(dim=24)
        case _:
            pytest.fail("unknown gate")


@pytest.fixture(params=["encoder", "decoder", "cross", "variable"])
def preset(
    request: pytest.FixtureRequest,
    placement: NormPlacement,
    ff_config: SwiGLUConfig | GEGLUConfig | HardSwishGLUConfig | GELUConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> Preset:
    """Build small two-layer presets, including a heterogeneous-width schedule."""
    match request.param:
        case "encoder":
            config = plain_encoder_config(24, 3, 2, norm_placement=placement)
            config.ff = ff_config
            model = TransformerEncoder(config)
        case "decoder":
            config = plain_decoder_config(24, 3, 2, norm_placement=placement)
            config.ff = ff_config
            model = TransformerDecoder(config)
        case "cross":
            config = plain_cross_attender_config(24, 3, 2, norm_placement=placement)
            config.ff = ff_config
            model = CrossAttender(config)
        case "variable":
            config = variable_width_encoder_config(
                d_models=[24, 48], dim_heads=8, norm_placement=placement
            )
            for layer in config.layers:
                layer.ff = ff_config.model_copy(update={"dim": layer.attn.dim})
            model = VariableWidthTransformerEncoder(config)
        case _:
            pytest.fail("unknown preset")
    device, dtype = device_dtype
    return model.to(device=device, dtype=dtype).eval()


@pytest.fixture
def preset_tree(preset: Preset) -> nn.ModuleDict:
    """Keep a preset under an unrelated parent so metadata synchronization is recursive too."""
    return nn.ModuleDict({"model": preset})


def test_preset_config_and_weights_roundtrip(
    preset: Preset,
    preset_tree: nn.ModuleDict,
    inputs: tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput],
    tmp_path: Path,
) -> None:
    """Converted preset JSON reconstructs both checkpoint layouts with output parity."""
    config = preset.config
    original_config = config.model_dump_json()
    state = {name: value.clone() for name, value in preset.state_dict().items()}
    expected = _run(preset, inputs)
    assert to_packed(preset_tree) is preset_tree
    assert preset_tree["model"] is preset
    assert config.model_dump_json() == original_config
    torch.testing.assert_close(
        _run(preset, inputs), expected, atol=atol(expected.dtype), rtol=atol(expected.dtype)
    )
    packed_config = type(config).model_validate_json(preset.config.model_dump_json())
    match packed_config:
        case TransformerEncoderConfig():
            restored = TransformerEncoder(packed_config)
        case TransformerDecoderConfig():
            restored = TransformerDecoder(packed_config)
        case CrossAttenderConfig():
            restored = CrossAttender(packed_config)
        case VariableWidthTransformerEncoderConfig():
            restored = VariableWidthTransformerEncoder(packed_config)
        case _:
            assert_never(packed_config)
    restored.to(inputs[0].x).eval()
    path = tmp_path / "packed.pt"
    torch.save(preset.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(_run(restored, inputs), _run(preset, inputs), rtol=0, atol=0)
    assert to_unpacked(preset_tree) is preset_tree
    assert preset.config.model_dump_json() == original_config
    _assert_state_equal(preset, state)
    torch.testing.assert_close(_run(preset, inputs), expected, rtol=0, atol=0)


def test_mixed_tree(preset: Preset) -> None:
    """A model with some manually packed layers can be converted and called repeatedly."""
    reference = deepcopy(preset)
    for name, child in preset.named_modules():
        if type(child) is SwiGLU:
            preset.set_submodule(name, to_packed(child))
            break
    assert to_packed(preset) is preset
    parameters = tuple(preset.parameters())
    assert to_packed(preset) is preset
    assert all(a is b for a, b in zip(preset.parameters(), parameters, strict=True))
    to_unpacked(preset)
    _assert_state_equal(preset, dict(reference.state_dict()))
