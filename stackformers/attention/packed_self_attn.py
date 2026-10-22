"""Opt-in packed projections for SelfAttention; ordinary attention stays unchanged."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from typing_extensions import override

from stackformers._projection_packing import load_packed_state, load_unpacked_state
from stackformers._result import unwrap_or_raise
from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import PackedSelfAttentionConfig, SelfAttentionConfig
from stackformers.attention.masking import SelfAttentionMask, resolve_self_mask
from stackformers.attention.ops import (
    _repeat_padded_kv_heads,
    packed_attn_or_fallback,
    padded_sdpa,
)
from stackformers.attention.self_attn import SelfAttention
from stackformers.sequence import (
    AllValid,
    PackedInput,
    PackedSequence,
    PaddedInput,
    SequenceInput,
    packed_to_padded,
)

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.attention.protocols import AttnBias
    from stackformers.positional.protocols import PosEncoding


_SEPARATE = ("to_q.weight", "to_k.weight", "to_v.weight")
_PACKED = "qkv.weight"


class PackedSelfAttention(nn.Module):
    """Share the original equation while packing input projection weights.

    Use ``from_unpacked`` to preserve existing weights and ``unpack`` to restore
    the original checkpoint layout. Convert before creating an optimizer.
    """

    def __init__(
        self,
        config: PackedSelfAttentionConfig,
        pos_encoding: PosEncoding,
        attn_bias: AttnBias | None = None,
    ) -> None:
        """Build packed projections and shared attention collaborators."""
        super().__init__()
        self.config = config
        h, dh = config.heads, config.dim_head
        widths = (
            config.heads * config.dim_head,
            config.effective_kv_heads * config.dim_head,
            config.effective_kv_heads * config.dim_head,
        )
        self.projection_widths = widths
        self.qkv = nn.Linear(config.dim, sum(widths), bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_encoding = pos_encoding
        self.attn_bias = NoAttnBias() if attn_bias is None else attn_bias
        self.q_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        nn.init.normal_(self.to_out.weight, std=0.02)

    def _project_qkv(self, x: Float[Tensor, "... d"]) -> tuple[Tensor, Tensor, Tensor]:
        """Project once and retain the semantic output order."""
        projected: Tensor = self.qkv(x)
        q, k, v = projected.split(self.projection_widths, dim=-1)
        return q, k, v

    @classmethod
    def from_unpacked(cls, source: SelfAttention) -> PackedSelfAttention:
        """Copy an ordinary module without retaining or changing its parameters."""
        config = PackedSelfAttentionConfig.model_validate(
            source.config.model_dump(exclude={"kind"})
        )
        with torch.device("meta"):
            target = cls(config, deepcopy(source.pos_encoding), deepcopy(source.attn_bias))
        load_packed_state(source, target, _SEPARATE, _PACKED)
        return target

    def unpack(self) -> SelfAttention:
        """Return an independent module compatible with ordinary checkpoints."""
        with torch.device("meta"):
            target = SelfAttention(
                SelfAttentionConfig.model_validate(self.config.model_dump(exclude={"kind"})),
                deepcopy(self.pos_encoding),
                deepcopy(self.attn_bias),
            )
        load_unpacked_state(self, target, _PACKED, _SEPARATE)
        return target

    def _forward_padded(self, input: PaddedInput, self_attn_mask: SelfAttentionMask) -> Tensor:
        """Attend using the packed projection in this sequence layout."""
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x = input.x
        projected_q, projected_k, projected_v = self._project_qkv(x)
        q = self.q_norm(rearrange(projected_q, "b n (h d) -> b h n d", h=h))
        k = self.k_norm(rearrange(projected_k, "b n (h d) -> b h n d", h=kv_h))
        v = rearrange(projected_v, "b n (h d) -> b h n d", h=kv_h)
        if groups > 1:
            k = _repeat_padded_kv_heads(k, groups)
            v = _repeat_padded_kv_heads(v, groups)
        q, k = self.pos_encoding.forward_padded(q, k, input.abs_positions, input.abs_positions)
        match input.mask:
            case AllValid():
                if type(self.attn_bias) is not NoAttnBias:
                    raise ValueError("all-valid attention requires NoAttnBias")
                bias = None
            case Tensor():
                bias = self.attn_bias(input)
        causal, window, allowed = unwrap_or_raise(
            resolve_self_mask(
                self_attn_mask, cfg.causal, cfg.window_size, input.x.shape[1], input.x.device
            )
        )
        out = padded_sdpa(q, k, v, input.mask, causal, window, bias, allowed)
        projected: Tensor = self.dropout(self.to_out(rearrange(out, "b h n d -> b n (h d)")))
        return projected

    def _forward_packed(self, input: PackedInput, self_attn_mask: SelfAttentionMask) -> Tensor:
        """Attend using the packed projection in this sequence layout."""
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x = input.x
        projected_q, projected_k, projected_v = self._project_qkv(x)
        q = self.q_norm(rearrange(projected_q, "nt (h d) -> nt h d", h=h))
        k = self.k_norm(rearrange(projected_k, "nt (h d) -> nt h d", h=kv_h))
        v = rearrange(projected_v, "nt (h d) -> nt h d", h=kv_h)
        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)
        q, k = self.pos_encoding.forward_packed(q, k, input.abs_positions, input.abs_positions)
        match self.attn_bias:
            case NoAttnBias():
                bias = None
            case _:
                bias = self.attn_bias(packed_to_padded(input))
        seq = PackedSequence(cu_seqlens=input.cu_seqlens, max_seqlen=input.max_seqlen)
        causal, window, allowed = unwrap_or_raise(
            resolve_self_mask(
                self_attn_mask, cfg.causal, cfg.window_size, input.max_seqlen, input.x.device
            )
        )
        out = packed_attn_or_fallback(q, k, v, seq, seq, causal, window, bias, allowed)
        projected: Tensor = self.dropout(self.to_out(rearrange(out, "nt h d -> nt (h d)")))
        return projected

    @override
    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> Tensor:
        """Attend with per-call visibility; explicit masks override causal/window defaults."""
        match input:
            case PaddedInput():
                return self._forward_padded(input, self_attn_mask)
            case PackedInput():
                return self._forward_packed(input, self_attn_mask)

    if TYPE_CHECKING:
        __call__ = forward
