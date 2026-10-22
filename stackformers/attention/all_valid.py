"""Mask-free cached attention using the existing projection and positional equations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from stackformers.attention.cache import CrossAttentionKVCache
from stackformers.attention.cached import _positioned_cross_query, _positioned_qkv
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.ops import _repeat_padded_kv_heads
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.sequence import AllValidInput

Attention = SelfAttention | PackedSelfAttention | CrossAttention | PackedCrossAttention
SelfAttentionModule = SelfAttention | PackedSelfAttention


def project_output(attention: Attention, output: Float[Tensor, "b h n dh"]) -> Tensor:
    """Reuse the owned output projection and dropout without query masking."""
    merged = rearrange(output, "b h n dh -> b n (h dh)")
    projected: Tensor = attention.dropout(attention.to_out(merged))
    return projected


def attend(
    attention: Attention,
    q: Float[Tensor, "b h n dh"],
    k: Float[Tensor, "b kv_h s dh"],
    v: Float[Tensor, "b kv_h s dh"],
) -> Tensor:
    """Attend to valid keys; one cached query must see every past/current position."""
    if k.shape[2] == 0:
        raise ValueError("all-valid attention requires at least one key")
    if attention.config.groups > 1:
        k = _repeat_padded_kv_heads(k, attention.config.groups)
        v = _repeat_padded_kv_heads(v, attention.config.groups)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    return project_output(attention, output)


def functional_self_attention(
    attention: SelfAttentionModule,
    input: AllValidInput,
    cache: CrossAttentionKVCache,
) -> tuple[Tensor, CrossAttentionKVCache]:
    """Append one positioned key and raw value without mutating the incoming cache."""
    q, k, v = _positioned_qkv(attention, input.x, input.abs_positions)
    updated = CrossAttentionKVCache(
        k=torch.cat((cache.k, k), dim=2), v=torch.cat((cache.v, v), dim=2)
    )
    return attend(attention, q, updated.k, updated.v), updated


def cross_attention(
    attention: CrossAttention | PackedCrossAttention,
    input: AllValidInput,
    cache: CrossAttentionKVCache,
) -> Tensor:
    """Project only the current query against an immutable, all-valid context cache."""
    q = _positioned_cross_query(attention, input.x, input.abs_positions)
    return attend(attention, q, cache.k, cache.v)
