from __future__ import annotations

import warnings

import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.ops import packed_attn_or_fallback, padded_sdpa, varlen_supported
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PackedSequence, PaddedInput, SequenceInput


class SelfAttention(nn.Module):
    """Multi-head self-attention.

    window_size=None (default) → global attention.
    window_size=w             → sliding-window attention with width w.

    Pass PaddedInput for inference or export, PackedInput for training — same weights.
    Falls back to padded SDPA when varlen_attn is unavailable (CPU, float32, torch.export).
    """

    def __init__(self, config: SelfAttentionConfig, pos_encoding: PosEncoding) -> None:
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

    def _forward_padded(self, input: PaddedInput) -> Tensor:
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x = input.x
        q = self.q_norm(rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h))
        k = self.k_norm(rearrange(self.to_k(x), "b n (h d) -> b h n d", h=kv_h))
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=kv_h)
        if groups > 1:
            k = repeat(k, "b h n d -> b (h g) n d", g=groups)
            v = repeat(v, "b h n d -> b (h g) n d", g=groups)
        q, k = self.pos_encoding.forward_padded(q, k, input.abs_positions, input.abs_positions)
        dropout_p = cfg.dropout if self.training else 0.0
        out = padded_sdpa(q, k, v, input.mask, cfg.causal, cfg.window_size, dropout_p)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))

    def _forward_packed(self, input: PackedInput) -> Tensor:
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x = input.x
        q = self.q_norm(rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h))
        k = self.k_norm(rearrange(self.to_k(x), "nt (h d) -> nt h d", h=kv_h))
        v = rearrange(self.to_v(x), "nt (h d) -> nt h d", h=kv_h)
        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)
        q, k = self.pos_encoding.forward_packed(q, k, input.abs_positions, input.abs_positions)
        seq = PackedSequence(cu_seqlens=input.cu_seqlens, max_seqlen=input.max_seqlen)
        if self.training and cfg.dropout > 0.0 and varlen_supported(q):
            warnings.warn(
                "dropout is not applied for PackedInput — varlen_attn does not support it.",
                UserWarning,
                stacklevel=2,
            )
        dropout_p = cfg.dropout if self.training else 0.0
        out = packed_attn_or_fallback(q, k, v, seq, seq, cfg.causal, cfg.window_size, dropout_p)
        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))

    def forward(self, input: SequenceInput) -> Tensor:
        match input:
            case PaddedInput():
                return self._forward_padded(input)
            case PackedInput():
                return self._forward_packed(input)
