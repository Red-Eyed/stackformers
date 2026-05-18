from __future__ import annotations

import warnings

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.ops import packed_attn_or_fallback, padding_mask, varlen_supported
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PackedSequence, PaddedInput, SequenceInput


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from x, keys/values from context.

    Always global (no windowing). Pass PaddedInput for inference or export, PackedInput for
    training. Falls back to padded SDPA when varlen_attn is unavailable (CPU, float32,
    torch.export).
    """

    def __init__(self, config: CrossAttentionConfig, pos_encoding: PosEncoding) -> None:
        super().__init__()
        self.config = config
        h, kv_h, dh = config.heads, config.effective_kv_heads, config.dim_head
        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.pos_encoding = pos_encoding
        self.q_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        nn.init.normal_(self.to_out.weight, std=0.02)

    def _forward_padded(self, x_input: PaddedInput, ctx_input: PaddedInput) -> Tensor:
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x, context = x_input.x, ctx_input.x
        q = self.q_norm(rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h))
        k = self.k_norm(rearrange(self.to_k(context), "b s (h d) -> b h s d", h=kv_h))
        v = rearrange(self.to_v(context), "b s (h d) -> b h s d", h=kv_h)
        if groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=groups)
        q, k = self.pos_encoding.forward_padded(
            q, k, x_input.abs_positions, ctx_input.abs_positions
        )
        attn_mask = padding_mask(ctx_input.mask, q.dtype)
        dropout_p = cfg.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return out * x_input.mask.unsqueeze(-1)

    def _forward_packed(self, x_input: PackedInput, ctx_input: PackedInput) -> Tensor:
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x, context = x_input.x, ctx_input.x
        q = self.q_norm(rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h))
        k = self.k_norm(rearrange(self.to_k(context), "nt (h d) -> nt h d", h=kv_h))
        v = rearrange(self.to_v(context), "nt (h d) -> nt h d", h=kv_h)
        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)
        q, k = self.pos_encoding.forward_packed(
            q, k, x_input.abs_positions, ctx_input.abs_positions
        )
        x_seq = PackedSequence(cu_seqlens=x_input.cu_seqlens, max_seqlen=x_input.max_seqlen)
        ctx_seq = PackedSequence(cu_seqlens=ctx_input.cu_seqlens, max_seqlen=ctx_input.max_seqlen)
        if self.training and cfg.dropout > 0.0 and varlen_supported(q):
            warnings.warn(
                "dropout is not applied for PackedInput — varlen_attn does not support it.",
                UserWarning,
                stacklevel=2,
            )
        dropout_p = cfg.dropout if self.training else 0.0
        out = packed_attn_or_fallback(
            q, k, v, x_seq, ctx_seq, causal=False, window_size=None, dropout_p=dropout_p
        )
        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        match x_input:
            case PaddedInput():
                return self._forward_padded(x_input, ctx_input)  # type: ignore[arg-type]
            case PackedInput():
                return self._forward_packed(x_input, ctx_input)  # type: ignore[arg-type]
