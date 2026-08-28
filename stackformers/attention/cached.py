"""Composable one-token attention executors with externally managed K/V tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.cache import CrossAttentionKVCache
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.ops import padded_sdpa
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PaddedInput, PaddedSequence


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


class CachedSelfAttentionWrapper(nn.Module):
    """Add growing one-token K/V execution around an existing self-attention module.

    The wrapped module owns every parameter. The incoming cache is required and stores
    position-encoded K plus raw V at KV-head width. Its token dimension must equal ``step_i``.
    Parameters and train/eval state are shared with the wrapped attention module.

    Example:
        ``output, next_cache = CachedSelfAttentionWrapper(attention)(token, cache, step_i)``
    """

    def __init__(self, attention: SelfAttention) -> None:
        """Validate the supported cached equation and retain the original attention weights."""
        super().__init__()
        if not attention.config.causal or attention.config.window_size is not None:
            raise NotImplementedError(
                "cached self-attention currently supports global causal attention only"
            )
        if not isinstance(attention.attn_bias, NoAttnBias):
            raise NotImplementedError("cached self-attention currently supports NoAttnBias only")
        self.attention = attention
        self.train(attention.training)

    def forward(
        self,
        input: PaddedInput,
        cache: Tensor,
        step_i: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode exactly one token and return output plus the one-position-longer cache."""
        if input.x.shape[1] != 1:
            raise ValueError("cached self-attention requires exactly one target token")

        attention = self.attention
        config = attention.config
        heads = config.heads
        kv_heads = config.effective_kv_heads
        q = attention.q_norm(rearrange(attention.to_q(input.x), "b n (h d) -> b h n d", h=heads))
        k = attention.k_norm(rearrange(attention.to_k(input.x), "b n (h d) -> b h n d", h=kv_heads))
        v = rearrange(attention.to_v(input.x), "b n (h d) -> b h n d", h=kv_heads)
        q, k = attention.pos_encoding.forward_padded(
            q,
            k,
            input.abs_positions,
            input.abs_positions,
        )

        cache_k, cache_v = cache.unbind(dim=0)
        updated_k = torch.cat((cache_k, k), dim=2)
        updated_v = torch.cat((cache_v, v), dim=2)
        attended_k = updated_k
        attended_v = updated_v
        if config.groups > 1:
            attended_k = repeat(attended_k, "b h s d -> b (h g) s d", g=config.groups)
            attended_v = repeat(attended_v, "b h s d -> b (h g) s d", g=config.groups)

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
        return out * input.mask.unsqueeze(-1), updated_cache


class CachedCrossAttentionWrapper(nn.Module):
    """Add reusable-context execution around an existing cross-attention module.

    Parameters and train/eval state are shared with the wrapped attention module.

    Example:
        ``wrapper = CachedCrossAttentionWrapper(attention)``
        ``output = wrapper(target, wrapper.build_cache(context), context_sequence)``
    """

    def __init__(self, attention: CrossAttention) -> None:
        """Retain the original attention module so cached and ordinary paths share weights."""
        super().__init__()
        self.attention = attention
        self.train(attention.training)

    def build_cache(self, context: PaddedInput) -> CrossAttentionKVCache:
        """Project and position one context while retaining configured KV-head width."""
        attention = self.attention
        kv_heads = attention.config.effective_kv_heads
        k = attention.k_norm(
            rearrange(attention.to_k(context.x), "b s (h d) -> b h s d", h=kv_heads)
        )
        v = rearrange(attention.to_v(context.x), "b s (h d) -> b h s d", h=kv_heads)
        k = _position_key(attention.pos_encoding, k, context.abs_positions)
        return CrossAttentionKVCache(k=k, v=v)

    def forward(
        self,
        input: PaddedInput,
        cache: CrossAttentionKVCache,
        context: PaddedSequence,
    ) -> Tensor:
        """Attend one padded target input to a preprojected fixed context."""
        attention = self.attention
        config = attention.config
        q = attention.q_norm(
            rearrange(attention.to_q(input.x), "b n (h d) -> b h n d", h=config.heads)
        )
        q = _position_query(attention.pos_encoding, q, input.abs_positions)
        k, v = cache
        if config.groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=config.groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=config.groups)
        out = padded_sdpa(
            q,
            k,
            v,
            context.mask,
            causal=False,
            window_size=None,
            bias=None,
        )
        out = attention.dropout(attention.to_out(rearrange(out, "b h n d -> b n (h d)")))
        return out * input.mask.unsqueeze(-1)
