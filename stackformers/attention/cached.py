"""Composable one-token attention executors with externally managed K/V tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from typing_extensions import override

from stackformers._result import unwrap_or_raise
from stackformers.attention.cache import CrossAttentionKVCache
from stackformers.attention.cache_validation import (
    cacheable_cross_attention,
    single_token,
    supported_self_attention,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.ops import _repeat_padded_kv_heads, mask_query_output, padded_sdpa
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.positional.protocols import PosEncoding
    from stackformers.sequence import PaddedSequence, PaddedTensor


def _project_qkv(
    attention: SelfAttention | PackedSelfAttention, x: Float[Tensor, "b n d"]
) -> tuple[Tensor, Tensor, Tensor]:
    """Reuse the owned projection layout without copying or unpacking weights."""
    match attention:
        case SelfAttention():
            return attention.to_q(x), attention.to_k(x), attention.to_v(x)
        case PackedSelfAttention():
            projected: Tensor = attention.qkv(x)
            q, k, v = projected.split(attention.projection_widths, dim=-1)
            return q, k, v
        case _:
            assert_never(attention)


def _project_kv(
    attention: CrossAttention | PackedCrossAttention, x: Float[Tensor, "b s d"]
) -> tuple[Tensor, Tensor]:
    """Project a fixed context once using separate K/V or a packed KV matrix."""
    match attention:
        case CrossAttention():
            return attention.to_k(x), attention.to_v(x)
        case PackedCrossAttention():
            projected: Tensor = attention.kv(x)
            k, v = projected.chunk(2, dim=-1)
            return k, v
        case _:
            assert_never(attention)


def _position_query(encoding: PosEncoding, q: Tensor, positions: Tensor) -> Tensor:
    """Position Q through the existing joint API without materializing a non-empty K."""
    empty_k = q[:, :, :0]
    empty_positions = positions[:, :0]
    positioned_q, _ = encoding.forward_padded(
        q,
        empty_k,
        positions,
        empty_positions,
    )
    return positioned_q


def _position_key(encoding: PosEncoding, k: Tensor, positions: Tensor) -> Tensor:
    """Position K through the existing joint API without materializing a non-empty Q."""
    empty_q = k[:, :, :0]
    empty_positions = positions[:, :0]
    _, positioned_k = encoding.forward_padded(
        empty_q,
        k,
        empty_positions,
        positions,
    )
    return positioned_k


def _positioned_qkv(
    attention: SelfAttention | PackedSelfAttention,
    x: Float[Tensor, "b n d"],
    positions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project, normalize and position new self keys and queries with owned modules."""
    config = attention.config
    projected_q, projected_k, projected_v = _project_qkv(attention, x)
    q = attention.q_norm(rearrange(projected_q, "b n (h d) -> b h n d", h=config.heads))
    k = attention.k_norm(
        rearrange(projected_k, "b n (h d) -> b h n d", h=config.effective_kv_heads)
    )
    v = rearrange(projected_v, "b n (h d) -> b h n d", h=config.effective_kv_heads)
    q, k = attention.pos_encoding.forward_padded(q, k, positions, positions)
    return q, k, v


def _positioned_cross_kv(
    attention: CrossAttention | PackedCrossAttention,
    x: Float[Tensor, "b s d"],
    positions: Tensor,
) -> CrossAttentionKVCache:
    """Build immutable normalized and positioned context keys and raw values."""
    projected_k, projected_v = _project_kv(attention, x)
    heads = attention.config.effective_kv_heads
    k = attention.k_norm(rearrange(projected_k, "b s (h d) -> b h s d", h=heads))
    v = rearrange(projected_v, "b s (h d) -> b h s d", h=heads)
    k = _position_key(attention.pos_encoding, k, positions)
    return CrossAttentionKVCache(k=k, v=v)


def _positioned_cross_query(
    attention: CrossAttention | PackedCrossAttention,
    x: Float[Tensor, "b n d"],
    positions: Tensor,
) -> Tensor:
    """Apply the same query projection, normalization and RoPE as ordinary attention."""
    q = attention.q_norm(
        rearrange(attention.to_q(x), "b n (h d) -> b h n d", h=attention.config.heads)
    )
    return _position_query(attention.pos_encoding, q, positions)


class CachedSelfAttentionWrapper(nn.Module):
    """Add growing one-token K/V execution around an existing self-attention module.

    The wrapped module owns every parameter. The incoming cache is required and stores
    position-encoded K plus raw V at KV-head width. Its token dimension must equal ``step_i``.
    Ordinary and packed projections share this cache layout. Parameters and train/eval state
    are shared with the wrapped attention module.
    Only exact built-in attention types are admitted; unsupported subclasses raise TypeError
    rather than silently replacing their forward behavior with the built-in cached equation.

    Example:
        ``output, next_cache = CachedSelfAttentionWrapper(attention)(token, cache, step_i)``
    """

    def __init__(self, attention: SelfAttention | PackedSelfAttention) -> None:
        """Validate the supported cached equation and retain the original attention weights."""
        super().__init__()
        self.attention = unwrap_or_raise(supported_self_attention(attention))
        self.train(attention.training)

    @override
    def forward(
        self,
        input: PaddedTensor,
        cache: Tensor,
        step_i: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode exactly one token and return output plus the one-position-longer cache."""
        unwrap_or_raise(single_token(input.x.shape[1]))

        attention = self.attention
        config = attention.config
        q, k, v = _positioned_qkv(attention, input.x, input.abs_positions)

        cache_k, cache_v = cache.unbind(dim=0)
        updated_k = torch.cat((cache_k, k), dim=2)
        updated_v = torch.cat((cache_v, v), dim=2)
        attended_k = updated_k
        attended_v = updated_v
        if config.groups > 1:
            attended_k = _repeat_padded_kv_heads(attended_k, config.groups)
            attended_v = _repeat_padded_kv_heads(attended_v, config.groups)

        cache_positions = torch.arange(updated_k.shape[2], device=cache.device)
        cache_mask = cache_positions.unsqueeze(0) <= step_i
        cache_mask = cache_mask.expand(input.x.shape[0], -1)
        out = padded_sdpa(
            q,
            attended_k,
            attended_v,
            cache_mask,
            causal=False,
            window_size=None,
            bias=None,
        )
        out = attention.dropout(attention.to_out(rearrange(out, "b h n d -> b n (h d)")))
        updated_cache = torch.stack((updated_k, updated_v), dim=0)
        return mask_query_output(out, input), updated_cache

    if TYPE_CHECKING:
        __call__ = forward


class CachedCrossAttentionWrapper(nn.Module):
    """Add reusable-context execution around an existing cross-attention module.

    Ordinary and packed KV projections use the same cache record. Parameters and train/eval
    state are shared with the wrapped attention module.
    Unsupported subclasses raise TypeError before context projections are evaluated.

    Example:
        ``wrapper = CachedCrossAttentionWrapper(attention)``
        ``output = wrapper(target, wrapper.build_cache(context), context_sequence)``
    """

    def __init__(self, attention: CrossAttention | PackedCrossAttention) -> None:
        """Retain the original attention module so cached and ordinary paths share weights."""
        super().__init__()
        self.attention = unwrap_or_raise(cacheable_cross_attention(attention))
        self.train(attention.training)

    def build_cache(self, context: PaddedTensor) -> CrossAttentionKVCache:
        """Project and position one context while retaining configured KV-head width."""
        attention = self.attention
        return _positioned_cross_kv(attention, context.x, context.abs_positions)

    @override
    def forward(
        self,
        input: PaddedTensor,
        cache: CrossAttentionKVCache,
        context: PaddedTensor | PaddedSequence,
    ) -> Tensor:
        """Attend one padded target input to a preprojected fixed context."""
        attention = self.attention
        config = attention.config
        q = _positioned_cross_query(attention, input.x, input.abs_positions)
        k, v = cache
        if config.groups > 1:
            k = _repeat_padded_kv_heads(k, config.groups)
            v = _repeat_padded_kv_heads(v, config.groups)
        out = padded_sdpa(
            q,
            k,
            v,
            context.mask,
            causal=False,
            window_size=None,
            bias=None,
        )
        projected: Tensor = attention.dropout(
            attention.to_out(rearrange(out, "b h n d -> b n (h d)"))
        )
        return mask_query_output(projected, input)

    if TYPE_CHECKING:
        __call__ = forward
