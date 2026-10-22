"""Opt-in packed projections for CrossAttention; ordinary attention stays unchanged."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from typing_extensions import override

from stackformers._projection_packing import load_packed_state, load_unpacked_state
from stackformers._result import unwrap_or_raise
from stackformers.attention.config import CrossAttentionConfig, PackedCrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.layout import matching_layouts
from stackformers.attention.masking import CrossAttentionMask, validate_cross_mask
from stackformers.attention.ops import (
    _repeat_padded_kv_heads,
    dense_sdpa,
    mask_query_output,
    packed_attn_or_fallback,
)
from stackformers.sequence import (
    AllValidInput,
    PackedInput,
    PackedSequence,
    PaddedInput,
    SequenceInput,
)

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.positional.protocols import PosEncoding


_SEPARATE = ("to_k.weight", "to_v.weight")
_PACKED = "kv.weight"


class PackedCrossAttention(nn.Module):
    """Share the original equation while packing input projection weights.

    Use ``from_unpacked`` to preserve existing weights and ``unpack`` to restore
    the original checkpoint layout. Convert before creating an optimizer.
    """

    def __init__(self, config: PackedCrossAttentionConfig, pos_encoding: PosEncoding) -> None:
        """Build packed projections and shared attention collaborators."""
        super().__init__()
        self.config = config
        h, dh = config.heads, config.dim_head
        widths = (config.effective_kv_heads * config.dim_head,) * 2
        self.to_q = nn.Linear(config.dim, config.heads * config.dim_head, bias=False)
        self.kv = nn.Linear(config.dim, sum(widths), bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_encoding = pos_encoding
        self.q_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        nn.init.normal_(self.to_out.weight, std=0.02)

    def _project_kv(self, context: Float[Tensor, "... d"]) -> tuple[Tensor, Tensor]:
        """Project once and retain the semantic output order."""
        projected: Tensor = self.kv(context)
        k, v = projected.chunk(2, dim=-1)
        return k, v

    @classmethod
    def from_unpacked(cls, source: CrossAttention) -> PackedCrossAttention:
        """Copy an ordinary module without retaining or changing its parameters."""
        config = PackedCrossAttentionConfig.model_validate(
            source.config.model_dump(exclude={"kind"})
        )
        with torch.device("meta"):
            target = cls(config, deepcopy(source.pos_encoding))
        load_packed_state(source, target, _SEPARATE, _PACKED)
        return target

    def unpack(self) -> CrossAttention:
        """Return an independent module compatible with ordinary checkpoints."""
        with torch.device("meta"):
            target = CrossAttention(
                CrossAttentionConfig.model_validate(self.config.model_dump(exclude={"kind"})),
                deepcopy(self.pos_encoding),
            )
        load_unpacked_state(self, target, _PACKED, _SEPARATE)
        return target

    def _forward_padded(
        self,
        x_input: PaddedInput | AllValidInput,
        ctx_input: PaddedInput | AllValidInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend using the packed projection in this sequence layout."""
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x, context = x_input.x, ctx_input.x
        projected_k, projected_v = self._project_kv(context)
        q = self.q_norm(rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h))
        k = self.k_norm(rearrange(projected_k, "b s (h d) -> b h s d", h=kv_h))
        v = rearrange(projected_v, "b s (h d) -> b h s d", h=kv_h)
        if groups > 1:
            k = _repeat_padded_kv_heads(k, groups)
            v = _repeat_padded_kv_heads(v, groups)
        q, k = self.pos_encoding.forward_padded(
            q, k, x_input.abs_positions, ctx_input.abs_positions
        )
        allowed = unwrap_or_raise(
            validate_cross_mask(cross_attn_mask, x.shape[1], context.shape[1], x.device)
        )
        out = dense_sdpa(
            q,
            k,
            v,
            ctx_input,
            causal=False,
            window_size=None,
            bias=None,
            allowed_pairs=allowed,
        )
        projected: Tensor = self.dropout(self.to_out(rearrange(out, "b h n d -> b n (h d)")))
        return mask_query_output(projected, x_input)

    def _forward_packed(
        self,
        x_input: PackedInput,
        ctx_input: PackedInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend using the packed projection in this sequence layout."""
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x, context = x_input.x, ctx_input.x
        projected_k, projected_v = self._project_kv(context)
        q = self.q_norm(rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h))
        k = self.k_norm(rearrange(projected_k, "nt (h d) -> nt h d", h=kv_h))
        v = rearrange(projected_v, "nt (h d) -> nt h d", h=kv_h)
        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)
        q, k = self.pos_encoding.forward_packed(
            q, k, x_input.abs_positions, ctx_input.abs_positions
        )
        x_seq = PackedSequence(cu_seqlens=x_input.cu_seqlens, max_seqlen=x_input.max_seqlen)
        ctx_seq = PackedSequence(cu_seqlens=ctx_input.cu_seqlens, max_seqlen=ctx_input.max_seqlen)
        allowed = unwrap_or_raise(
            validate_cross_mask(cross_attn_mask, x_input.max_seqlen, ctx_input.max_seqlen, x.device)
        )
        out = packed_attn_or_fallback(
            q,
            k,
            v,
            x_seq,
            ctx_seq,
            causal=False,
            window_size=None,
            bias=None,
            allowed_pairs=allowed,
        )
        projected: Tensor = self.dropout(self.to_out(rearrange(out, "nt h d -> nt (h d)")))
        return projected

    @overload
    def forward(
        self,
        x_input: AllValidInput,
        ctx_input: AllValidInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend between explicitly all-valid dense sequences."""
        ...

    @overload
    def forward(
        self,
        x_input: PaddedInput,
        ctx_input: PaddedInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend between padded inputs with independently sized sequence axes."""
        ...

    @overload
    def forward(
        self,
        x_input: PackedInput,
        ctx_input: PackedInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend between packed inputs with matching document counts."""
        ...

    @override
    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend within one shared layout; raise ValueError for mixed layouts."""
        match unwrap_or_raise(matching_layouts(x_input, ctx_input)):
            case PaddedInput() as queries, PaddedInput() as context:
                return self._forward_padded(queries, context, cross_attn_mask=cross_attn_mask)
            case AllValidInput() as queries, AllValidInput() as context:
                return self._forward_padded(queries, context, cross_attn_mask=cross_attn_mask)
            case PackedInput() as queries, PackedInput() as context:
                return self._forward_packed(queries, context, cross_attn_mask=cross_attn_mask)

    if TYPE_CHECKING:
        # Preserve nn.Module's hook machinery at runtime while exposing the
        # same layout contract for module calls and explicit forward calls.
        __call__ = forward
