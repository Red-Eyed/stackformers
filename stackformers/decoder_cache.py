"""Composition-only decoder executors for externally managed attention caches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from stackformers.attention.cache import (
    CrossAttentionKVCache,
    DecoderCrossAttentionCache,
    DecoderStepOutput,
)
from stackformers.attention.cached import (
    CachedCrossAttentionWrapper,
    CachedSelfAttentionWrapper,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import (
    Decoder,
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.sequence import PaddedInput, PaddedSequence


def _unwrap_decoder(module: nn.Module) -> Decoder:
    """Resolve a low-level decoder from either a stack or its existing preset wrapper."""
    candidate = module if isinstance(module, Decoder) else getattr(module, "_decoder", module)
    if not isinstance(candidate, Decoder):
        raise TypeError(f"{type(module).__name__} does not contain a Stackformers Decoder")
    return candidate


def _self_attention(module: object) -> SelfAttention:
    """Require the standard self-attention whose projections the cache executor reuses."""
    if not isinstance(module, SelfAttention):
        raise TypeError(f"cached decoding does not support {type(module).__name__} self-attention")
    return module


def _cross_attention(module: object) -> CrossAttention:
    """Require the standard cross-attention whose projections the cache executor reuses."""
    if not isinstance(module, CrossAttention):
        raise TypeError(f"cached decoding does not support {type(module).__name__} cross-attention")
    return module


def _layer_cross_cache(cache: Tensor) -> CrossAttentionKVCache:
    """Expose one dense decoder-layer cache through the attention-level named record."""
    return CrossAttentionKVCache(k=cache[0], v=cache[1])


class CachedDecoderLayerBase(nn.Module, ABC):
    """Cache executor components shared by every decoder normalization topology."""

    def __init__(self, layer: DecoderLayerBase) -> None:
        """Wrap the layer attentions while sharing its feed-forward parameters."""
        super().__init__()
        self.self_attn = CachedSelfAttentionWrapper(_self_attention(layer.self_attn))
        self.cross_attn = CachedCrossAttentionWrapper(_cross_attention(layer.cross_attn))
        self.ff = layer.ff

    @abstractmethod
    def forward(
        self,
        input: PaddedInput,
        cross_cache: Tensor,
        self_cache: Tensor,
        context: PaddedSequence,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Decode one token and return this layer's extended self cache."""


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
        cross_cache: Tensor,
        self_cache: Tensor,
        context: PaddedSequence,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Mirror the ordinary pre-norm residual equation using cached attention."""
        normed_self = input._replace(x=self.norm_self(input.x))
        self_out, updated_self_cache = self.self_attn(normed_self, self_cache, step_i)
        x = input.x + self_out
        input = input._replace(x=x)
        normed_cross = input._replace(x=self.norm_cross(input.x))
        x = input.x + self.cross_attn(normed_cross, _layer_cross_cache(cross_cache), context)
        x = x + self.ff(self.norm_ff(x))
        return input._replace(x=x), updated_self_cache


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
        cross_cache: Tensor,
        self_cache: Tensor,
        context: PaddedSequence,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Mirror the ordinary post-norm residual equation using cached attention."""
        self_out, updated_self_cache = self.self_attn(input, self_cache, step_i)
        x = self.norm_self(input.x + self_out)
        input = input._replace(x=x)
        x = self.norm_cross(x + self.cross_attn(input, _layer_cross_cache(cross_cache), context))
        x = self.norm_ff(x + self.ff(x))
        return input._replace(x=x), updated_self_cache


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
        cross_cache: Tensor,
        self_cache: Tensor,
        context: PaddedSequence,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Mirror the ordinary sandwich-norm residual equation using cached attention."""
        normed_self = input._replace(x=self.norm_self_pre(input.x))
        self_out, updated_self_cache = self.self_attn(normed_self, self_cache, step_i)
        x = input.x + self.norm_self_post(self_out)
        input = input._replace(x=x)
        normed_cross = input._replace(x=self.norm_cross_pre(input.x))
        x = x + self.norm_cross_post(
            self.cross_attn(normed_cross, _layer_cross_cache(cross_cache), context)
        )
        x = x + self.norm_ff_post(self.ff(self.norm_ff_pre(x)))
        return input._replace(x=x), updated_self_cache


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
        cross_cache: Tensor,
        self_cache: Tensor,
        context: PaddedSequence,
        step_i: Tensor,
    ) -> tuple[PaddedInput, Tensor]:
        """Mirror the ordinary reordered residual equation using cached attention."""
        self_out, updated_self_cache = self.self_attn(input, self_cache, step_i)
        x = input.x + self.norm_self(self_out)
        input = input._replace(x=x)
        x = x + self.norm_cross(self.cross_attn(input, _layer_cross_cache(cross_cache), context))
        x = x + self.norm_ff(self.ff(x))
        return input._replace(x=x), updated_self_cache


def _cached_layer(layer: DecoderLayerBase) -> CachedDecoderLayerBase:
    """Build the cache executor matching one existing normalization topology."""
    match layer:
        case DecoderLayer():
            return CachedDecoderLayer(layer)
        case PostNormDecoderLayer():
            return CachedPostNormDecoderLayer(layer)
        case SandwichNormDecoderLayer():
            return CachedSandwichNormDecoderLayer(layer)
        case ReorderedNormDecoderLayer():
            return CachedReorderedNormDecoderLayer(layer)
        case _:
            raise TypeError(f"cached decoding does not support {type(layer).__name__}")


def _decoder_layers(decoder: Decoder) -> tuple[DecoderLayerBase, ...]:
    """Validate and expose the ordinary decoder's registered layer sequence."""
    layers: list[DecoderLayerBase] = []
    for layer in decoder.layers:
        if not isinstance(layer, DecoderLayerBase):
            raise TypeError(f"expected DecoderLayerBase, received {type(layer).__name__}")
        layers.append(layer)
    return tuple(layers)


class DecoderCrossAttentionCacheBuilder(nn.Module):
    """Exportable once-per-context K/V projection composed from an existing decoder.

    Cross-attention parameters and train/eval state remain shared with the ordinary decoder.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Wrap only each decoder layer's cross-attention module and share its weights."""
        super().__init__()
        source = _unwrap_decoder(decoder)
        self.layers = nn.ModuleList(
            CachedCrossAttentionWrapper(_cross_attention(layer.cross_attn))
            for layer in _decoder_layers(source)
        )
        self.train(source.training)

    def forward(self, context: PaddedInput) -> DecoderCrossAttentionCache:
        """Return dense per-layer cross K/V plus the context validity mask."""
        layer_caches = tuple(
            cast(CachedCrossAttentionWrapper, layer).build_cache(context) for layer in self.layers
        )
        kv = torch.stack(
            tuple(torch.stack((cache.k, cache.v), dim=0) for cache in layer_caches),
            dim=0,
        )
        return DecoderCrossAttentionCache(
            kv=kv,
            context=PaddedSequence(mask=context.mask),
        )


class CachedDecoderWrapper(nn.Module):
    """One-token cached executor composed from an unchanged existing decoder stack.

    Every parameter and train/eval state remains shared with the ordinary decoder. Calling
    ``train()`` or ``eval()`` on either view therefore affects the same underlying modules.

    Example:
        ``cache_builder = DecoderCrossAttentionCacheBuilder(decoder)``
        ``wrapper = CachedDecoderWrapper(decoder)``
        ``result = wrapper(token, cache_builder(context), self_cache, step_i)``
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Build topology-specific executors that share the ordinary decoder's parameters."""
        super().__init__()
        source = _unwrap_decoder(decoder)
        self.layers = nn.ModuleList(_cached_layer(layer) for layer in _decoder_layers(source))
        self.final_norm = source.final_norm
        self.train(source.training)

    def forward(
        self,
        input: PaddedInput,
        cross_cache: DecoderCrossAttentionCache,
        self_kv_cache: Tensor,
        step_i: Tensor,
    ) -> DecoderStepOutput:
        """Decode one token and return its output plus the one-position-longer self cache."""
        if cross_cache.kv.shape[0] != len(self.layers):
            raise ValueError(
                f"cross cache has {cross_cache.kv.shape[0]} layers, decoder has {len(self.layers)}"
            )
        if self_kv_cache.shape[0] != len(self.layers):
            raise ValueError(
                f"self cache has {self_kv_cache.shape[0]} layers, decoder has {len(self.layers)}"
            )

        updated_layer_caches: list[Tensor] = []
        for layer_i, layer in enumerate(self.layers):
            input, updated_layer_cache = layer(
                input,
                cross_cache.kv[layer_i],
                self_kv_cache[layer_i],
                cross_cache.context,
                step_i,
            )
            updated_layer_caches.append(updated_layer_cache)
        return DecoderStepOutput(
            x=self.final_norm(input.x),
            self_kv_cache=torch.stack(updated_layer_caches, dim=0),
        )
