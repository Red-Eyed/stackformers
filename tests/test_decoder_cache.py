"""Tests for decoder-level cross-attention caching and export adapters."""

from __future__ import annotations

import pytest
import torch

from stackformers.attention.cache import DecoderCrossAttentionCache
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder
from stackformers.decoder_cache import (
    CachedDecoderWrapper,
    DecoderCrossAttentionCacheBuilder,
)
from stackformers.norm.config import NormPlacement
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.presets.decoder import TransformerDecoder, plain_decoder_config
from stackformers.sequence import PaddedInput, make_padded_input
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run

B, N, S, D, H = 2, 4, 6, 64, 1


def _empty_self_cache(decoder: Decoder, target: PaddedInput) -> torch.Tensor:
    """Allocate the required zero-length self K/V tensor for the first decoder call."""
    return target.x.new_zeros(len(decoder.layers), 2, target.x.shape[0], H, 0, D // H)


def _build_decoder(norm_placement: NormPlacement = "pre") -> Decoder:
    """Build the low-level decoder owned by the public uniform preset."""
    preset = TransformerDecoder(
        plain_decoder_config(
            dim=D,
            heads=H,
            num_layers=2,
            norm_placement=norm_placement,
        )
    )
    return preset._decoder


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


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_cached_forward_matches_existing_forward(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    norm_placement: NormPlacement,
) -> None:
    """Every normalization topology has exact full-prefix/one-token output parity."""
    target, context = decoder_inputs
    decoder = _build_decoder(norm_placement).eval()
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


def test_decoder_cache_preserves_state_dict_contract() -> None:
    """Composing cache executors leaves ordinary parameters and checkpoint keys unchanged."""
    decoder = _build_decoder()
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
    assert cache.kv.shape == (2, 2, B, H, S, D // H)


def test_cache_support_does_not_add_methods_to_existing_classes() -> None:
    """Ordinary model classes retain their stable method surface after cache support is imported."""
    assert not hasattr(SelfAttention, "forward_cached")
    assert not hasattr(CrossAttention, "build_cache")
    assert not hasattr(CrossAttention, "forward_cached")
    assert not hasattr(Decoder, "build_cross_attention_cache")
    assert not hasattr(Decoder, "forward_cached")
    assert not hasattr(TransformerDecoder, "build_cross_attention_cache")
    assert not hasattr(TransformerDecoder, "forward_cached")


def test_transformer_decoder_exposes_cached_path(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """Cache wrappers compose the public preset without changing its existing API."""
    target, context = decoder_inputs
    decoder = TransformerDecoder(plain_decoder_config(dim=D, heads=H, num_layers=2)).eval()
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


def test_decoder_cached_forward_supports_cross_attention_rope(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
) -> None:
    """The decoder positions cross K once and cross Q on every cached step."""
    target, context = decoder_inputs
    decoder = _build_decoder()
    for layer in decoder.layers:
        assert isinstance(layer.cross_attn, CrossAttention)
        layer.cross_attn.pos_encoding = RotaryEmbedding1D(RoPE1DConfig(dim_head=D // H))
    decoder.eval()
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
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """The once-per-context cache builder supports static and dynamic ONNX export."""
    _, context = decoder_inputs
    decoder = _build_decoder()
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
def test_cached_decoder_exports_to_onnx(
    decoder_inputs: tuple[PaddedInput, PaddedInput],
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Required dense caches export with dynamic batch and context length."""
    target, context = decoder_inputs
    decoder = _build_decoder()
    cached_decoder = CachedDecoderWrapper(decoder)
    cache_builder = DecoderCrossAttentionCacheBuilder(decoder)
    cross_cache = cache_builder(context)
    self_cache = target.x.new_zeros(2, 2, B, H, 2, D // H)
    target_token = PaddedInput(*(tensor[:, :1] for tensor in target))
    step_i = torch.tensor([2], dtype=torch.int64)
    resized_target = make_padded_input(torch.randn(1, 1, D), torch.ones(1, 1, dtype=torch.bool))
    resized_context = make_padded_input(
        torch.randn(1, S - 2, D),
        torch.ones(1, S - 2, dtype=torch.bool),
    )
    resized_cross_cache = cache_builder(resized_context)
    resized_self_cache = resized_target.x.new_zeros(2, 2, 1, H, 0, D // H)
    resized_step_i = torch.tensor([0], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1, max=B)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=S)
    past_tokens = torch.export.Dim("past_tokens", min=0, max=N)
    shapes = torch.export.ShapesCollection()
    for tensor in target_token:
        shapes[tensor] = {0: batch}
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
