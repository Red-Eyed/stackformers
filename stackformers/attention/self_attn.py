from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.attention.varlen import varlen_attn as _varlen_attn

from stackformers.attention.config import SelfAttentionConfig
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PackedSequence, PaddedInput, SequenceInput

# ── padded helpers ────────────────────────────────────────────────────────────


def _padding_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias.masked_fill_(~mask, torch.finfo(dtype).min)
    return bias.view(mask.shape[0], 1, 1, mask.shape[1])


def _window_mask(n: int, s: int, window_size: int, causal: bool, device: torch.device) -> Tensor:
    """Additive sliding-window mask (0 = attend, -inf = ignore): (1, 1, n, s)."""
    q_pos = torch.arange(n, device=device).unsqueeze(1)
    k_pos = torch.arange(s, device=device).unsqueeze(0)
    if causal:
        allowed = (k_pos <= q_pos) & (k_pos >= q_pos - window_size)
    else:
        half = window_size // 2
        allowed = (k_pos >= q_pos - half) & (k_pos <= q_pos + half)
    mask = torch.zeros(1, 1, n, s, dtype=torch.float, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


# ── packed helper ─────────────────────────────────────────────────────────────


def _varlen_window(causal: bool, window_size: int | None) -> tuple[int, int]:
    if window_size is None:
        return (-1, 0) if causal else (-1, -1)
    return (window_size, 0) if causal else (window_size // 2, window_size // 2)


def _packed_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_seq: PackedSequence,
    k_seq: PackedSequence,
    causal: bool,
    window_size: int | None,
) -> Tensor:
    if not q.is_cuda:
        raise RuntimeError(f"packed attention requires CUDA tensors, got {q.device}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"packed attention requires float16 or bfloat16, got {q.dtype}")
    win = _varlen_window(causal, window_size)
    result = _varlen_attn(
        query=q,
        key=k,
        value=v,
        cu_seq_q=q_seq.cu_seqlens.to(torch.int32),
        cu_seq_k=k_seq.cu_seqlens.to(torch.int32),
        max_q=q_seq.max_seqlen,
        max_k=k_seq.max_seqlen,
        window_size=win,
    )
    assert isinstance(result, Tensor)
    return result


# ── module ────────────────────────────────────────────────────────────────────


class SelfAttention(nn.Module):
    """Multi-head self-attention.

    window_size=None (default) → global attention.
    window_size=w             → sliding-window attention with width w.

    Pass PaddedInput for inference, PackedInput for training — same weights.
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
        if cfg.window_size is None:
            attn_mask = _padding_mask(input.mask, q.dtype)
            if cfg.causal:
                n, s = q.shape[-2], k.shape[-2]
                attn_mask = attn_mask + _window_mask(n, s, s, causal=True, device=q.device)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            n, s = q.shape[-2], k.shape[-2]
            win_mask = _window_mask(n, s, cfg.window_size, cfg.causal, q.device)
            win_mask = win_mask + _padding_mask(input.mask, q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=win_mask, dropout_p=dropout_p)
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
        if self.training and cfg.dropout > 0.0:
            warnings.warn(
                "dropout is not applied for PackedInput — varlen_attn does not support it.",
                UserWarning,
                stacklevel=2,
            )
        out = _packed_attn(q, k, v, seq, seq, cfg.causal, cfg.window_size)
        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))

    def forward(self, input: SequenceInput) -> Tensor:
        match input:
            case PaddedInput():
                return self._forward_padded(input)
            case PackedInput():
                return self._forward_packed(input)
