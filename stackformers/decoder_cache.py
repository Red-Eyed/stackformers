"""Composition-only decoder executors for externally managed attention caches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, assert_never, cast

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
from stackformers.attention.cache import (
    CrossAttentionKVCache,
    DecoderCrossAttentionCache,
    DecoderStepOutput,
)
from stackformers.attention.cache_validation import (
    cacheable_cross_attention,
    cacheable_self_attention,
    matching_layer_counts,
    supported_self_attention,
)
from stackformers.attention.cached import (
    CachedCrossAttentionWrapper,
    CachedSelfAttentionWrapper,
)
from stackformers.decoder import (
    Decoder,
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.result import Err, Ok, Result
from stackformers.sequence import PaddedInput, PaddedSequence

if TYPE_CHECKING:
    from jaxtyping import Float


def _unwrap_decoder(module: nn.Module) -> Result[Decoder, TypeError]:
    """Resolve a low-level decoder from either a stack or its existing preset wrapper."""
    candidate = module if isinstance(module, Decoder) else getattr(module, "_decoder", module)
    if type(candidate) is not Decoder:
        return Err(TypeError(f"{type(module).__name__} does not contain a Stackformers Decoder"))
    return Ok(candidate)


class _CachedCrossContext(NamedTuple):
    """Keep projected keys/values inseparable from their context validity mask."""

    kv: CrossAttentionKVCache
    sequence: PaddedSequence


def _cross_context(
    cache: Float[Tensor, "two b kv_h s dh"] | None,
    context: PaddedSequence | None,
) -> Result[NoContext | WithContext[_CachedCrossContext], ValueError]:
    """Validate the legacy argument pair before entering the shared layer equation."""
    match cache, context:
        case None, None:
            return Ok(NoContext())
        case Tensor(), PaddedSequence():
            kv = CrossAttentionKVCache(k=cache[0], v=cache[1])
            return Ok(WithContext(_CachedCrossContext(kv=kv, sequence=context)))
        case _:
            return Err(ValueError("cross cache and context must both be present or both be absent"))


def _cached_cross_attention(
    attention: CachedCrossAttentionWrapper,
    input: PaddedInput,
    context: _CachedCrossContext,
) -> Tensor:
    """Adapt a complete cached context to the shared cross-attention branch contract."""
    return attention(input, context.kv, context.sequence)


def _supported_layer(
    layer: DecoderLayerBase,
) -> Result[
    DecoderLayer | PostNormDecoderLayer | SandwichNormDecoderLayer | ReorderedNormDecoderLayer,
    TypeError,
]:
    """Admit built-in topologies without silently discarding subclass forward overrides."""
    match layer:
        case DecoderLayer() if type(layer) is DecoderLayer:
            return Ok(layer)
        case PostNormDecoderLayer() if type(layer) is PostNormDecoderLayer:
            return Ok(layer)
        case SandwichNormDecoderLayer() if type(layer) is SandwichNormDecoderLayer:
            return Ok(layer)
        case ReorderedNormDecoderLayer() if type(layer) is ReorderedNormDecoderLayer:
            return Ok(layer)
        case _:
            return Err(TypeError(f"cached decoding does not support {type(layer).__name__}"))


class CachedDecoderLayerBase(nn.Module, ABC):
    """Cache executor components shared by every decoder normalization topology."""

    def __init__(self, layer: DecoderLayerBase) -> None:
        """Wrap the layer attentions while sharing its feed-forward parameters."""
        super().__init__()
        unwrap_or_raise(_supported_layer(layer))
        self.self_attn = CachedSelfAttentionWrapper(
            unwrap_or_raise(cacheable_self_attention(layer.self_attn))
        )
        self.cross_attn = CachedCrossAttentionWrapper(
            unwrap_or_raise(cacheable_cross_attention(layer.cross_attn))
        )
        self.ff = layer.ff

    @abstractmethod
    @override
    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor | None,
        self_cache: Tensor,
        context: PaddedSequence | None,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Decode one token; reject half-present cross cache/context pairs with ValueError."""

    if TYPE_CHECKING:
        __call__ = forward


class CachedDecoderLayer(CachedDecoderLayerBase):
    """Cached executor for the existing pre-norm decoder topology."""

    def __init__(self, layer: DecoderLayer) -> None:
        """Share the pre-norm layer's attention, feed-forward, and norm modules."""
        super().__init__(layer)
        self.norm_self = layer.norm_self
        self.norm_cross = layer.norm_cross
        self.norm_ff = layer.norm_ff

    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor | None,
        self_cache: Tensor,
        context: PaddedSequence | None,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Run the shared pre-norm equation with validated cache state."""
        cross_context = unwrap_or_raise(_cross_context(cross_cache, context))
        return pre_norm_decoder(
            input,
            partial(self.self_attn, cache=self_cache, step_i=step_i),
            partial(_cached_cross_attention, self.cross_attn),
            cross_context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )

    if TYPE_CHECKING:
        __call__ = forward


class CachedPostNormDecoderLayer(CachedDecoderLayerBase):
    """Cached executor for the existing post-norm decoder topology."""

    def __init__(self, layer: PostNormDecoderLayer) -> None:
        """Share the post-norm layer's attention, feed-forward, and norm modules."""
        super().__init__(layer)
        self.norm_self = layer.norm_self
        self.norm_cross = layer.norm_cross
        self.norm_ff = layer.norm_ff

    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor | None,
        self_cache: Tensor,
        context: PaddedSequence | None,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Run the shared post-norm equation with validated cache state."""
        cross_context = unwrap_or_raise(_cross_context(cross_cache, context))
        return post_norm_decoder(
            input,
            partial(self.self_attn, cache=self_cache, step_i=step_i),
            partial(_cached_cross_attention, self.cross_attn),
            cross_context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )

    if TYPE_CHECKING:
        __call__ = forward


class CachedSandwichNormDecoderLayer(CachedDecoderLayerBase):
    """Cached executor for the existing sandwich-norm decoder topology."""

    def __init__(self, layer: SandwichNormDecoderLayer) -> None:
        """Share the sandwich layer's attention, feed-forward, and six norms."""
        super().__init__(layer)
        self.norm_self_pre = layer.norm_self_pre
        self.norm_self_post = layer.norm_self_post
        self.norm_cross_pre = layer.norm_cross_pre
        self.norm_cross_post = layer.norm_cross_post
        self.norm_ff_pre = layer.norm_ff_pre
        self.norm_ff_post = layer.norm_ff_post

    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor | None,
        self_cache: Tensor,
        context: PaddedSequence | None,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Run the shared sandwich-norm equation with validated cache state."""
        cross_context = unwrap_or_raise(_cross_context(cross_cache, context))
        return sandwich_norm_decoder(
            input,
            partial(self.self_attn, cache=self_cache, step_i=step_i),
            partial(_cached_cross_attention, self.cross_attn),
            cross_context,
            self.ff,
            self.norm_self_pre,
            self.norm_self_post,
            self.norm_cross_pre,
            self.norm_cross_post,
            self.norm_ff_pre,
            self.norm_ff_post,
        )

    if TYPE_CHECKING:
        __call__ = forward


class CachedReorderedNormDecoderLayer(CachedDecoderLayerBase):
    """Cached executor for the existing residual-post-norm decoder topology."""

    def __init__(self, layer: ReorderedNormDecoderLayer) -> None:
        """Share the reordered layer's attention, feed-forward, and output norms."""
        super().__init__(layer)
        self.norm_self = layer.norm_self
        self.norm_cross = layer.norm_cross
        self.norm_ff = layer.norm_ff

    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor | None,
        self_cache: Tensor,
        context: PaddedSequence | None,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Run the shared reordered-norm equation with validated cache state."""
        cross_context = unwrap_or_raise(_cross_context(cross_cache, context))
        return reordered_norm_decoder(
            input,
            partial(self.self_attn, cache=self_cache, step_i=step_i),
            partial(_cached_cross_attention, self.cross_attn),
            cross_context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )

    if TYPE_CHECKING:
        __call__ = forward


def _cached_layer(
    layer: DecoderLayerBase,
) -> Result[CachedDecoderLayerBase, TypeError | NotImplementedError]:
    """Validate cache admission before constructing an executor, without exception recovery."""
    supported_layer = _supported_layer(layer)
    match supported_layer:
        case Ok(value=layer):
            pass
        case Err() as failure:
            return failure
        case _:
            assert_never(supported_layer)
    admission = _cached_attention_admission(layer)
    match admission:
        case Err() as failure:
            return failure
        case Ok():
            pass
        case _:
            assert_never(admission)
    match layer:
        case DecoderLayer():
            return Ok(CachedDecoderLayer(layer))
        case PostNormDecoderLayer():
            return Ok(CachedPostNormDecoderLayer(layer))
        case SandwichNormDecoderLayer():
            return Ok(CachedSandwichNormDecoderLayer(layer))
        case ReorderedNormDecoderLayer():
            return Ok(CachedReorderedNormDecoderLayer(layer))
        case _:
            assert_never(layer)


def _cached_attention_admission(
    layer: DecoderLayerBase,
) -> Result[tuple[()], TypeError | NotImplementedError]:
    """Reject unsupported attention types and cache equations before invoking constructors."""
    supported = supported_self_attention(layer.self_attn)
    match supported:
        case Err() as failure:
            return failure
        case Ok():
            pass
        case _:
            assert_never(supported)
    cross_attention = cacheable_cross_attention(layer.cross_attn)
    match cross_attention:
        case Err() as failure:
            return failure
        case Ok():
            return Ok(())
        case _:
            assert_never(cross_attention)


def _decoder_layers(decoder: Decoder) -> Result[tuple[DecoderLayerBase, ...], TypeError]:
    """Validate and expose the ordinary decoder's registered layer sequence."""
    layers: list[DecoderLayerBase] = []
    for layer in decoder.layers:
        if not isinstance(layer, DecoderLayerBase):
            return Err(TypeError(f"expected DecoderLayerBase, received {type(layer).__name__}"))
        layers.append(layer)
    return Ok(tuple(layers))


class DecoderCrossAttentionCacheBuilder(nn.Module):
    """Exportable once-per-context K/V projection composed from an existing decoder.

    Cross-attention parameters and train/eval state remain shared with the ordinary decoder.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Wrap only each decoder layer's cross-attention module and share its weights."""
        super().__init__()
        source = unwrap_or_raise(_unwrap_decoder(decoder))
        self.layers = nn.ModuleList(
            CachedCrossAttentionWrapper(
                unwrap_or_raise(cacheable_cross_attention(layer.cross_attn))
            )
            for layer in unwrap_or_raise(_decoder_layers(source))
        )
        self.train(source.training)

    @override
    def forward(self, context: PaddedInput) -> DecoderCrossAttentionCache:
        """Return dense per-layer cross K/V plus the context validity mask."""
        layer_caches = tuple(
            cast("CachedCrossAttentionWrapper", layer).build_cache(context) for layer in self.layers
        )
        kv = torch.stack(
            tuple(torch.stack((cache.k, cache.v), dim=0) for cache in layer_caches),
            dim=0,
        )
        return DecoderCrossAttentionCache(
            kv=kv,
            context=PaddedSequence(mask=context.mask),
        )

    if TYPE_CHECKING:
        __call__ = forward


class CachedDecoderWrapper(nn.Module):
    """One-token cached executor composed from an unchanged existing decoder stack.

    Every parameter and train/eval state remains shared with the ordinary decoder. Calling
    ``train()`` or ``eval()`` on either view therefore affects the same underlying modules.
    Only the built-in Decoder stack and four built-in layer topologies are supported;
    subclasses raise TypeError because their overridden equations cannot be inferred safely.

    Example:
        ``cache_builder = DecoderCrossAttentionCacheBuilder(decoder)``
        ``wrapper = CachedDecoderWrapper(decoder)``
        ``result = wrapper(token, cache_builder(context), self_cache, step_i)``
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Build topology-specific executors that share the ordinary decoder's parameters."""
        super().__init__()
        source = unwrap_or_raise(_unwrap_decoder(decoder))
        self.layers = nn.ModuleList(
            unwrap_or_raise(_cached_layer(layer))
            for layer in unwrap_or_raise(_decoder_layers(source))
        )
        self.final_norm = source.final_norm
        self.train(source.training)

    @override
    def forward(
        self,
        input: PaddedInput,
        cross_cache: DecoderCrossAttentionCache | None,
        self_kv_cache: Tensor,
        step_i: Tensor,
    ) -> DecoderStepOutput:
        """Decode one token; a None cross cache skips cross-attention and its norms."""
        unwrap_or_raise(
            matching_layer_counts(
                len(self.layers) if cross_cache is None else cross_cache.kv.shape[0],
                self_kv_cache.shape[0],
                len(self.layers),
            )
        )

        updated_layer_caches: list[Tensor] = []
        for layer_i, layer in enumerate(self.layers):
            input, updated_layer_cache = layer(
                input,
                None if cross_cache is None else cross_cache.kv[layer_i],
                self_kv_cache[layer_i],
                None if cross_cache is None else cross_cache.context,
                step_i,
            )
            updated_layer_caches.append(updated_layer_cache)
        return DecoderStepOutput(
            x=self.final_norm(input.x),
            self_kv_cache=torch.stack(updated_layer_caches, dim=0),
        )

    if TYPE_CHECKING:
        __call__ = forward
