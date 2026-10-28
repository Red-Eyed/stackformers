"""Explicit all-valid single-sequence decoding with functional per-layer caches."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple, TypeVar, assert_never, cast

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers._decoder_equations import (
    NoContext,
    WithContext,
    post_norm_decoder,
    pre_norm_decoder,
    reordered_norm_decoder,
    sandwich_norm_decoder,
)
from stackformers._result import unwrap_or_raise
from stackformers.attention.all_valid import (
    SelfAttentionModule,
    cross_attention,
    functional_self_attention,
)
from stackformers.attention.cache import CrossAttentionKVCache
from stackformers.attention.cache_validation import (
    cacheable_cross_attention,
    supported_self_attention,
)
from stackformers.attention.cached import _positioned_cross_kv
from stackformers.decoder import (
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.decoder_cache import _decoder_layers, _supported_layer, _unwrap_decoder
from stackformers.result import Err, Ok, Result
from stackformers.sequence import AllValid, PaddedTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float

    from stackformers.attention.cross_attn import CrossAttention
    from stackformers.attention.packed_cross_attn import PackedCrossAttention

StateT = TypeVar("StateT")
LayerCaches = tuple[CrossAttentionKVCache, ...]
SupportedLayer = (
    DecoderLayer | PostNormDecoderLayer | SandwichNormDecoderLayer | ReorderedNormDecoderLayer
)


class AllValidDecoderStepOutput(NamedTuple):
    """Decoder embeddings and independently owned, growing K/V caches per layer."""

    x: Float[Tensor, "b one d"]
    self_kv_cache: LayerCaches


def _admit_decoder(decoder: nn.Module) -> tuple[nn.ModuleList, nn.Module]:
    """Validate built-in topology and cached equations before retaining source modules."""
    source = unwrap_or_raise(_unwrap_decoder(decoder))
    layers = unwrap_or_raise(_decoder_layers(source))
    for layer in layers:
        unwrap_or_raise(_supported_layer(layer))
        unwrap_or_raise(supported_self_attention(layer.self_attn))
        unwrap_or_raise(cacheable_cross_attention(layer.cross_attn))
    return nn.ModuleList(layers), cast("nn.Module", source.final_norm)


def _allocate_self_cache(layers: nn.ModuleList, capacity: int, dtype: torch.dtype) -> LayerCaches:
    """Allocate separate K/V tensors using each self-attention layer's own geometry/device."""
    buffers: list[CrossAttentionKVCache] = []
    for layer in layers:
        attention = cast("SelfAttentionModule", cast("DecoderLayerBase", layer).self_attn)
        config = attention.config
        shape = (1, config.effective_kv_heads, capacity, config.dim_head)
        device = next(attention.parameters()).device
        buffers.append(
            CrossAttentionKVCache(
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
        )
    return tuple(buffers)


def _validate_token(input: PaddedTensor) -> Result[tuple[()], TypeError | ValueError]:
    """Reject padding records, multi-sequence batches and multi-token decoder calls."""
    if not isinstance(input, PaddedTensor) or not isinstance(input.mask, AllValid):
        return Err(TypeError("all-valid decoding requires PaddedTensor with mask=AllValid()"))
    if input.x.shape[0] != 1 or input.x.shape[1] != 1:
        return Err(ValueError("all-valid decoding requires batch size one and exactly one token"))
    if input.abs_positions.shape[:2] != input.x.shape[:2]:
        return Err(ValueError("token coordinates must match the input batch and sequence"))
    return Ok(())


def _decode_layer(
    layer: DecoderLayerBase,
    input: PaddedTensor,
    self_attention: Callable[[PaddedTensor], tuple[Tensor, StateT]],
    context: CrossAttentionKVCache | None,
) -> tuple[PaddedTensor, StateT]:
    """Select the existing residual equation while substituting only attention execution."""
    cross = partial(
        cross_attention, cast("CrossAttention | PackedCrossAttention", layer.cross_attn)
    )
    cross_context = NoContext() if context is None else WithContext(context)
    match cast("SupportedLayer", layer):
        case DecoderLayer() as source:
            return pre_norm_decoder(
                input,
                self_attention,
                cross,
                cross_context,
                source.ff,
                source.norm_self,
                source.norm_cross,
                source.norm_ff,
            )
        case PostNormDecoderLayer() as source:
            return post_norm_decoder(
                input,
                self_attention,
                cross,
                cross_context,
                source.ff,
                source.norm_self,
                source.norm_cross,
                source.norm_ff,
            )
        case SandwichNormDecoderLayer() as source:
            return sandwich_norm_decoder(
                input,
                self_attention,
                cross,
                cross_context,
                source.ff,
                source.norm_self_pre,
                source.norm_self_post,
                source.norm_cross_pre,
                source.norm_cross_post,
                source.norm_ff_pre,
                source.norm_ff_post,
            )
        case ReorderedNormDecoderLayer() as source:
            return reordered_norm_decoder(
                input,
                self_attention,
                cross,
                cross_context,
                source.ff,
                source.norm_self,
                source.norm_cross,
                source.norm_ff,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _validate_layer_counts(
    layers: int, self_cache: LayerCaches, cross_cache: LayerCaches | None
) -> Result[tuple[()], ValueError]:
    """Reject incomplete cache trees before any decoder layer executes."""
    if len(self_cache) != layers or (cross_cache is not None and len(cross_cache) != layers):
        return Err(ValueError("cache layer counts must match the decoder"))
    return Ok(())


class AllValidCrossAttentionCacheBuilder(nn.Module):
    """Project all-valid encoder memory once, sharing the decoder's original parameters."""

    def __init__(self, decoder: nn.Module) -> None:
        """Retain admitted cross-attention modules without copying parameters or state."""
        super().__init__()
        layers, _ = _admit_decoder(decoder)
        self.layers = nn.ModuleList(
            cast("nn.Module", cast("DecoderLayerBase", layer).cross_attn) for layer in layers
        )
        # Recursive train() would reset the source modules' individual modes.
        self.training = decoder.training
        self.layers.training = decoder.training

    @override
    def forward(self, context: PaddedTensor) -> LayerCaches:
        """Return separate immutable K/V tensors; reject empty or multi-sequence memory."""
        if not isinstance(context, PaddedTensor) or not isinstance(context.mask, AllValid):
            raise TypeError(
                "all-valid cache construction requires PaddedTensor with mask=AllValid()"
            )
        if context.x.shape[0] != 1 or context.x.shape[1] == 0:
            raise ValueError("all-valid context requires one nonempty sequence")
        return tuple(
            _positioned_cross_kv(
                cast("CrossAttention | PackedCrossAttention", layer),
                context.x,
                context.abs_positions,
            )
            for layer in self.layers
        )

    if TYPE_CHECKING:
        __call__ = forward


class AllValidCachedDecoderWrapper(nn.Module):
    """Decode one valid token with functional caches and no padding-related operations.

    Every past/current self position and every context cache position must be valid.
    Empty self caches are supported. Coordinates come from the supplied token, not
    from cache length. Source parameters and training state remain shared; source
    checkpoints are unchanged. Incoming caches are never mutated.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Retain the admitted decoder layers and final normalization by reference."""
        super().__init__()
        self.layers, self.final_norm = _admit_decoder(decoder)
        # Recursive train() would reset the source modules' individual modes.
        self.training = decoder.training
        self.layers.training = decoder.training

    def empty_cache(self, *, dtype: torch.dtype) -> LayerCaches:
        """Create independent empty self caches at the intended projection dtype.

        Geometry and device come from each self-attention layer, independently
        of the cross-attention heads. Use the autocast dtype under autocast.
        """
        return _allocate_self_cache(self.layers, 0, dtype)

    @override
    def forward(
        self, input: PaddedTensor, cross_cache: LayerCaches | None, self_kv_cache: LayerCaches
    ) -> AllValidDecoderStepOutput:
        """Append one token to each layer cache and return the normalized decoder output."""
        unwrap_or_raise(_validate_token(input))
        unwrap_or_raise(_validate_layer_counts(len(self.layers), self_kv_cache, cross_cache))
        updated: list[CrossAttentionKVCache] = []
        for index, layer in enumerate(self.layers):
            source = cast("DecoderLayerBase", layer)
            attention = cast("SelfAttentionModule", source.self_attn)
            self_call = partial(functional_self_attention, attention, cache=self_kv_cache[index])
            context = None if cross_cache is None else cross_cache[index]
            input, cache = _decode_layer(source, input, self_call, context)
            updated.append(cache)
        return AllValidDecoderStepOutput(self.final_norm(input.x), tuple(updated))

    if TYPE_CHECKING:
        __call__ = forward
