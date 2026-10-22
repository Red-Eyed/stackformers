"""Tests for decoder-level cross-attention caching and export adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from stackformers.attention.cache import DecoderCrossAttentionCache
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer, DecoderLayerBase
from stackformers.decoder_cache import (
    CachedDecoderLayer,
    CachedDecoderLayerBase,
    CachedDecoderWrapper,
    DecoderCrossAttentionCacheBuilder,
    _cached_layer,
    _cross_context,
)
from stackformers.packing import to_packed, to_unpacked
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.presets.decoder import TransformerDecoder, plain_decoder_config
from stackformers.result import Err, Ok
from stackformers.sequence import PaddedInput, PaddedSequence, make_padded_input
from tests.conftest import atol
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run

if TYPE_CHECKING:
    from stackformers.norm.config import NormPlacement

B, N, S, D, H = 2, 4, 6, 64, 4


class CustomDecoderLayer(DecoderLayer):
    """Represent an extension whose full-layer behavior cannot be inferred by caching."""


class CustomDecoder(Decoder):
    """Represent an extension whose stack forward may differ from the built-in equation."""


def _empty_self_cache(decoder: Decoder, target: PaddedInput) -> torch.Tensor:
    """Allocate the required zero-length self K/V tensor for the first decoder call."""
    return target.x.new_zeros(
        len(decoder.layers), 2, target.x.shape[0], _kv_heads(decoder), 0, D // H
    )


def _build_decoder(norm_placement: NormPlacement = "pre", kv_heads: int = H) -> Decoder:
    """Build the low-level decoder owned by the public uniform preset."""
    config = plain_decoder_config(
        dim=D,
        heads=H,
        num_layers=2,
        norm_placement=norm_placement,
    )
    config = config.model_copy(
        update={
            "self_attn": config.self_attn.model_copy(update={"kv_heads": kv_heads}),
            "cross_attn": config.cross_attn.model_copy(update={"kv_heads": kv_heads}),
        }
    )
    preset = TransformerDecoder(config)
    return preset._decoder


@pytest.fixture(
    params=[(False, H), (True, H), (True, 2), (True, 1)],
    ids=["separate-mha", "packed-mha", "packed-gqa", "packed-mqa"],
)
def cache_decoder(request: pytest.FixtureRequest) -> Decoder:
    """Build ordinary MHA or packed MHA/GQA/MQA for cache and export contracts."""
    packed, kv_heads = request.param
    decoder = _build_decoder(kv_heads=kv_heads).eval()
    if packed:
        to_packed(decoder)
    return decoder


def _kv_heads(decoder: Decoder) -> int:
    """Read cache geometry from either supported projection layout."""
    attention = decoder.layers[0].self_attn
    assert isinstance(attention, (SelfAttention, PackedSelfAttention))
    return attention.config.effective_kv_heads


def _input_to(input: PaddedInput, device: torch.device, dtype: torch.dtype) -> PaddedInput:
    """Move model inputs while retaining the boolean mask dtype."""
    return PaddedInput(
        x=input.x.to(device=device, dtype=dtype),
        mask=input.mask.to(device=device),
        abs_positions=input.abs_positions.to(device=device, dtype=dtype),
    )


@pytest.fixture
def decoder_inputs() -> tuple[PaddedInput, PaddedInput]:
    """Build deterministic-shape padded target and context inputs."""
    target = make_padded_input(
        torch.randn(B, N, D),
        torch.ones(B, N, dtype=torch.bool),
    )
    context_mask = torch.ones(B, S, dtype=torch.bool)
    context_mask[1, -2:] = False
    context = make_padded_input(torch.randn(B, S, D), context_mask)
    return target, context


@pytest.fixture(params=["pre", "post", "sandwich", "reordered"])
def cached_layer(request: pytest.FixtureRequest) -> CachedDecoderLayerBase:
    """Construct every public cached topology for direct-call boundary validation."""
    decoder = _build_decoder(request.param)
    layer = decoder.layers[0]
    assert isinstance(layer, DecoderLayerBase)
    outcome = _cached_layer(layer)
    assert isinstance(outcome, Ok)
    return outcome.value


@pytest.fixture
def custom_decoder_layer() -> CustomDecoderLayer:
    """Retain supported collaborators so only subclass admission can reject this layer."""
    layer = _build_decoder().layers[0]
    assert isinstance(layer, DecoderLayer)
    return CustomDecoderLayer(
        layer.self_attn,
        layer.cross_attn,
        layer.ff,
        layer.norm_self,
        layer.norm_cross,
        layer.norm_ff,
    )


@pytest.fixture
def custom_decoder() -> CustomDecoder:
    """Retain built-in layers so only stack subclass admission can reject this decoder."""
    decoder = _build_decoder()
    layer = decoder.layers[0]
    assert isinstance(layer, DecoderLayerBase)
    return CustomDecoder([layer], decoder.final_norm)


@pytest.mark.parametrize("missing", ["cache", "context"])
def test_cached_layer_rejects_incomplete_cross_context(
    cached_layer: CachedDecoderLayerBase,
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half-present context fails before self-attention or any cache computation."""
    target, context = decoder_inputs
    cross_cache = torch.zeros(2, B, H, S, D // H) if missing != "cache" else None
    sequence = PaddedSequence(context.mask) if missing != "context" else None
    outcome = _cross_context(cross_cache, sequence)
    assert isinstance(outcome, Err)
    calls = Mock(side_effect=AssertionError("attention must not run before admission"))
    monkeypatch.setattr(cached_layer.self_attn, "forward", calls)
    token = PaddedInput(*(tensor[:, :1] for tensor in target))
    self_cache = torch.zeros(2, B, H, 0, D // H)
    with pytest.raises(ValueError) as caught:
        cached_layer(token, cross_cache, self_cache, sequence, torch.tensor([0]))
    assert caught.value.args == outcome.error.args
    calls.assert_not_called()


def test_custom_decoder_layer_rejected(custom_decoder_layer: CustomDecoderLayer) -> None:
    """Both direct cached-layer construction and decoder admission reject unknown equations."""
    outcome = _cached_layer(custom_decoder_layer)
    assert isinstance(outcome, Err)
    with pytest.raises(TypeError) as caught:
        CachedDecoderLayer(custom_decoder_layer)
    assert caught.value.args == outcome.error.args


@pytest.mark.parametrize("adapter", [CachedDecoderWrapper, DecoderCrossAttentionCacheBuilder])
def test_custom_decoder_stack_rejected(
    custom_decoder: CustomDecoder,
    adapter: type[CachedDecoderWrapper] | type[DecoderCrossAttentionCacheBuilder],
) -> None:
    """Neither deployment adapter may bypass an unknown decoder stack equation."""
    with pytest.raises(TypeError, match="does not contain a Stackformers Decoder"):
        adapter(custom_decoder)


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_decoder_cached_forward_matches_existing_forward(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    norm_placement: NormPlacement,
    with_context: bool,
    packed: bool,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Every normalization topology has exact full-prefix/one-token output parity."""
    target, context = decoder_inputs
    device, dtype = device_dtype
    target = _input_to(target, device, dtype)
    context = _input_to(context, device, dtype)
    decoder = _build_decoder(norm_placement).to(device=device, dtype=dtype).eval()
    if packed:
        to_packed(decoder)
    cache_builder = DecoderCrossAttentionCacheBuilder(decoder)
    cached_decoder = CachedDecoderWrapper(decoder)
    cross_cache = cache_builder(context) if with_context else None
    self_cache = _empty_self_cache(decoder, target)

    with torch.no_grad():
        for step in range(N):
            prefix = PaddedInput(*(tensor[:, : step + 1] for tensor in target))
            token = PaddedInput(*(tensor[:, step : step + 1] for tensor in target))
            expected = decoder(prefix, context if with_context else None)[:, -1:]
            result = cached_decoder(
                token,
                cross_cache,
                self_cache,
                torch.tensor([step], dtype=torch.int64, device=device),
            )
            torch.testing.assert_close(result.x, expected, atol=atol(dtype), rtol=atol(dtype))
            self_cache = result.self_kv_cache


def test_integrated_cached_model_survives_packing_round_trip(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Existing executors and live caches remain usable after registered model conversion."""
    device, dtype = device_dtype
    target, context = (_input_to(input, device, dtype) for input in decoder_inputs)
    decoder = _build_decoder(kv_heads=2).to(device=device, dtype=dtype).eval()
    builder = DecoderCrossAttentionCacheBuilder(decoder)
    cached = CachedDecoderWrapper(decoder)
    integrated = torch.nn.ModuleList([decoder, builder, cached])
    state = {name: value.clone() for name, value in integrated.state_dict().items()}
    cross_cache = builder(context)
    self_cache = _empty_self_cache(decoder, target)

    with torch.no_grad():
        for step, convert in enumerate((to_packed, to_unpacked)):
            prefix = PaddedInput(*(tensor[:, : step + 1] for tensor in target))
            token = PaddedInput(*(tensor[:, step : step + 1] for tensor in target))
            expected = decoder(prefix, context)[:, -1:]
            assert convert(integrated) is integrated
            layer = decoder.layers[0]
            cached_layer = cached.layers[0]
            assert isinstance(cached_layer, CachedDecoderLayerBase)
            assert cached_layer.self_attn.attention is layer.self_attn
            assert cached_layer.cross_attn.attention is layer.cross_attn
            assert builder.layers[0].attention is layer.cross_attn
            rebuilt_cache = builder(context)
            torch.testing.assert_close(
                rebuilt_cache.kv, cross_cache.kv, atol=atol(dtype), rtol=atol(dtype)
            )
            result = cached(token, cross_cache, self_cache, torch.tensor([step], device=device))
            torch.testing.assert_close(result.x, expected, atol=atol(dtype), rtol=atol(dtype))
            self_cache = result.self_kv_cache
            assert self_cache.shape == (2, 2, B, 2, step + 1, D // H)

    for name, value in integrated.state_dict().items():
        assert torch.equal(value, state[name])


def test_decoder_cache_preserves_state_dict_contract(cache_decoder: Decoder) -> None:
    """Composing cache executors leaves ordinary parameters and checkpoint keys unchanged."""
    decoder = cache_decoder
    keys_before = tuple(decoder.state_dict())
    parameter_count_before = sum(parameter.numel() for parameter in decoder.parameters())
    parameter_ids = {id(parameter) for parameter in decoder.parameters()}
    cached_decoder = CachedDecoderWrapper(decoder)
    cache_builder = DecoderCrossAttentionCacheBuilder(decoder)

    context = make_padded_input(
        torch.randn(B, S, D),
        torch.ones(B, S, dtype=torch.bool),
    )
    cache = cache_builder(context)

    assert tuple(decoder.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in decoder.parameters()) == parameter_count_before
    assert {id(parameter) for parameter in cached_decoder.parameters()} == parameter_ids
    assert {id(parameter) for parameter in cache_builder.parameters()} < parameter_ids
    assert cache._fields == ("kv", "context")
    assert cache.context._fields == ("mask",)
    assert cache.context.mask is context.mask
    assert cache.kv.shape == (2, 2, B, _kv_heads(decoder), S, D // H)


def test_cache_support_does_not_add_methods_to_existing_classes() -> None:
    """Ordinary model classes retain their stable method surface after cache support is imported."""
    assert not hasattr(SelfAttention, "forward_cached")
    assert not hasattr(CrossAttention, "build_cache")
    assert not hasattr(CrossAttention, "forward_cached")
    assert not hasattr(Decoder, "build_cross_attention_cache")
    assert not hasattr(Decoder, "forward_cached")
    assert not hasattr(TransformerDecoder, "build_cross_attention_cache")
    assert not hasattr(TransformerDecoder, "forward_cached")


@pytest.mark.parametrize("packed", [False, True])
def test_transformer_decoder_exposes_cached_path(
    packed: bool,
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """Cache wrappers compose the public preset without changing its existing API."""
    target, context = decoder_inputs
    decoder = TransformerDecoder(plain_decoder_config(dim=D, heads=H, num_layers=2)).eval()
    if packed:
        to_packed(decoder)
    builder = DecoderCrossAttentionCacheBuilder(decoder)
    cached_decoder = CachedDecoderWrapper(decoder)
    self_cache = target.x.new_zeros(2, 2, B, H, 0, D // H)
    token = PaddedInput(*(tensor[:, :1] for tensor in target))
    step_i = torch.tensor([0], dtype=torch.int64)

    with torch.no_grad():
        expected = decoder(token, context)
        cross_cache = builder(context)
        actual = cached_decoder(token, cross_cache, self_cache, step_i)

    torch.testing.assert_close(actual.x, expected)
    assert actual.self_kv_cache.shape == (*self_cache.shape[:4], 1, self_cache.shape[-1])
    assert not builder.training
    assert not cached_decoder.training


@pytest.mark.parametrize("packed", [False, True])
def test_decoder_cached_forward_supports_cross_attention_rope(
    packed: bool,
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """The decoder positions cross K once and cross Q on every cached step."""
    target, context = decoder_inputs
    decoder = _build_decoder()
    for layer in decoder.layers:
        assert isinstance(layer.cross_attn, CrossAttention)
        layer.cross_attn.pos_encoding = RotaryEmbedding1D(RoPE1DConfig(dim_head=D // H))
    decoder.eval()
    if packed:
        to_packed(decoder)
    cache_builder = DecoderCrossAttentionCacheBuilder(decoder)
    cached_decoder = CachedDecoderWrapper(decoder)
    cross_cache = cache_builder(context)
    self_cache = _empty_self_cache(decoder, target)

    with torch.no_grad():
        for step in range(N):
            prefix = PaddedInput(*(tensor[:, : step + 1] for tensor in target))
            token = PaddedInput(*(tensor[:, step : step + 1] for tensor in target))
            expected = decoder(prefix, context)[:, -1:]
            result = cached_decoder(
                token,
                cross_cache,
                self_cache,
                torch.tensor([step], dtype=torch.int64),
            )
            torch.testing.assert_close(result.x, expected)
            self_cache = result.self_kv_cache


def test_decoder_rejects_cache_for_different_layer_count(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """A cache cannot silently pair layer-specific projections with the wrong decoder."""
    target, context = decoder_inputs
    decoder = _build_decoder()
    cached_decoder = CachedDecoderWrapper(decoder)
    cache = DecoderCrossAttentionCacheBuilder(decoder)(context)
    invalid_cache = DecoderCrossAttentionCache(
        kv=cache.kv[:-1],
        context=cache.context,
    )
    self_cache = _empty_self_cache(decoder, target)

    with pytest.raises(ValueError, match="cross cache has 1 layers, decoder has 2"):
        cached_decoder(
            PaddedInput(*(tensor[:, :1] for tensor in target)),
            invalid_cache,
            self_cache,
            torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_decoder_cache_builder_exports_to_onnx(
    cache_decoder: Decoder,
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """The once-per-context cache builder supports static and dynamic ONNX export."""
    _, context = decoder_inputs
    decoder = cache_decoder
    builder = DecoderCrossAttentionCacheBuilder(decoder)
    resized_context = make_padded_input(
        torch.randn(1, S - 2, D),
        torch.ones(1, S - 2, dtype=torch.bool),
    )
    batch = torch.export.Dim("batch", min=1, max=B)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=S)
    shapes = torch.export.ShapesCollection()
    for tensor in context:
        shapes[tensor] = {0: batch, 1: context_tokens}

    export_and_run(
        builder,
        (context,),
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(builder, (context,)),
        runtime_args=(resized_context,),
    )


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
@pytest.mark.parametrize("with_context", [False, True])
def test_cached_decoder_exports_to_onnx(
    cache_decoder: Decoder,
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    shape_mode: ExportShapeMode,
    opset_version: int,
    with_context: bool,
) -> None:
    """Required dense caches export with dynamic batch and context length."""
    target, context = decoder_inputs
    decoder = cache_decoder
    cached_decoder = CachedDecoderWrapper(decoder)
    cache_builder = DecoderCrossAttentionCacheBuilder(decoder)
    cross_cache = cache_builder(context) if with_context else None
    self_cache = target.x.new_zeros(2, 2, B, _kv_heads(decoder), 2, D // H)
    target_token = PaddedInput(*(tensor[:, :1] for tensor in target))
    step_i = torch.tensor([2], dtype=torch.int64)
    resized_target = make_padded_input(torch.randn(1, 1, D), torch.ones(1, 1, dtype=torch.bool))
    resized_context = make_padded_input(
        torch.randn(1, S - 2, D),
        torch.ones(1, S - 2, dtype=torch.bool),
    )
    resized_cross_cache = cache_builder(resized_context) if with_context else None
    resized_self_cache = resized_target.x.new_zeros(2, 2, 1, _kv_heads(decoder), 0, D // H)
    resized_step_i = torch.tensor([0], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1, max=B)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=S)
    past_tokens = torch.export.Dim("past_tokens", min=0, max=N)
    shapes = torch.export.ShapesCollection()
    for tensor in target_token:
        shapes[tensor] = {0: batch}
    if cross_cache is not None:
        shapes[cross_cache.kv] = {2: batch, 4: context_tokens}
        shapes[cross_cache.context.mask] = {0: batch, 1: context_tokens}
    shapes[self_cache] = {2: batch, 4: past_tokens}

    export_and_run(
        cached_decoder,
        (target_token, cross_cache, self_cache, step_i),
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(
            cached_decoder,
            (target_token, cross_cache, self_cache, step_i),
        ),
        runtime_args=(
            resized_target,
            resized_cross_cache,
            resized_self_cache,
            resized_step_i,
        ),
    )


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
@pytest.mark.parametrize("packed", [False, True])
def test_context_free_cached_decoder_compiles_and_exports(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    norm_placement: NormPlacement,
    packed: bool,
) -> None:
    """Cached execution exports without context inputs or cross projections."""
    target, _ = decoder_inputs
    decoder = _build_decoder(norm_placement).eval()
    if packed:
        to_packed(decoder)
    cached = CachedDecoderWrapper(decoder)
    token = PaddedInput(*(tensor[:, :1] for tensor in target))
    cache = _empty_self_cache(decoder, target)
    args = (token, None, cache, torch.tensor([0], dtype=torch.int64))
    expected = cached(*args)
    compiled = torch.compile(cached, backend="eager", fullgraph=True)
    actual = compiled(*args)
    torch.testing.assert_close(actual.x, expected.x)
    torch.testing.assert_close(actual.self_kv_cache, expected.self_kv_cache)
    exported = torch.export.export(cached, args).module()
    actual = exported(*args)
    torch.testing.assert_close(actual.x, expected.x)
    torch.testing.assert_close(actual.self_kv_cache, expected.self_kv_cache)


def test_context_free_decoder_rejects_wrong_self_cache_layer_count(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """Skipping cross-cache validation retains self-cache admission checks."""
    target, _ = decoder_inputs
    decoder = _build_decoder()
    cached = CachedDecoderWrapper(decoder)
    token = PaddedInput(*(tensor[:, :1] for tensor in target))
    with pytest.raises(ValueError, match="self cache has 1 layers, decoder has 2"):
        cached(token, None, _empty_self_cache(decoder, target)[:-1], torch.tensor([0]))
