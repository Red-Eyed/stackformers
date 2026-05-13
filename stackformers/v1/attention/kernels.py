from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


class SDPAKernel(nn.Module):
    """Padded-sequence SDPA kernel using torch.nn.functional.scaled_dot_product_attention.

    Supports causal masking, attention bias, and key-padding mask.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        v: Float[Tensor, "b h s dh"],
        attn_mask: Float[Tensor, "b 1 n s"] | None,
        attn_bias: Float[Tensor, "h n s"] | None,
        is_causal: bool,
    ) -> Float[Tensor, "b h n dh"]:
        if attn_bias is not None:
            bias = attn_bias.unsqueeze(0)  # (1, h, n, s)
            combined = bias + (attn_mask if attn_mask is not None else 0.0)
        else:
            combined = attn_mask

        dropout_p = self.dropout if self.training else 0.0

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined,
            dropout_p=dropout_p,
            is_causal=is_causal and combined is None,
        )


try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    _HAS_VARLEN_ATTN = True
except ImportError:
    _HAS_VARLEN_ATTN = False


class VarlenSDPAKernel(nn.Module):
    """Packed variable-length SDPA kernel (full attention, no windowing).

    Primary path: torch.nn.attention.varlen.varlen_attn — a single fused kernel
    call, requires CUDA and a half-precision dtype (fp16 / bf16).

    Fallback: per-sequence loop over F.scaled_dot_product_attention — always
    available, used automatically on CPU and float32 (e.g. in tests).
    """

    def __init__(self, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = dropout
        # (-1, 0) = causal (attend to past + present), (-1, -1) = full bidirectional
        self._window_size = (-1, 0) if causal else (-1, -1)

    def forward(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        v: Float[Tensor, "nt h dh"],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Float[Tensor, "nt h dh"]:
        if _HAS_VARLEN_ATTN and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            cu = cu_seqlens.to(torch.int32)
            result = _varlen_attn(
                query=q,
                key=k,
                value=v,
                cu_seq_q=cu,
                cu_seq_k=cu,
                max_q=max_seqlen,
                max_k=max_seqlen,
                window_size=self._window_size,
            )
            assert isinstance(result, Tensor)
            return result

        # CPU / float32 fallback — loop over each sequence in the pack
        batch = cu_seqlens.shape[0] - 1
        outputs: list[Tensor] = []
        for i in range(batch):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            qi = rearrange(q[start:end], "n h d -> 1 h n d")
            ki = rearrange(k[start:end], "s h d -> 1 h s d")
            vi = rearrange(v[start:end], "s h d -> 1 h s d")
            dropout_p = self.dropout if self.training else 0.0
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=self.causal
            )
            outputs.append(rearrange(out_i, "1 h n d -> n h d"))
        return torch.cat(outputs, dim=0)


def _build_window_mask(
    n: int,
    s: int,
    window_size: int,
    causal: bool,
    device: torch.device,
) -> Float[Tensor, "1 1 n s"]:
    """Build an additive sliding-window mask (0 = keep, -inf = mask out).

    Causal: each query i attends to keys in [i - window_size, i].
    Bidirectional: each query i attends to keys in [i - half, i + half].
    """
    q_pos = torch.arange(n, device=device).unsqueeze(1)  # (n, 1)
    k_pos = torch.arange(s, device=device).unsqueeze(0)  # (1, s)
    if causal:
        allowed = (k_pos <= q_pos) & (k_pos >= q_pos - window_size)
    else:
        half = window_size // 2
        allowed = (k_pos >= q_pos - half) & (k_pos <= q_pos + half)
    mask = torch.zeros(1, 1, n, s, dtype=torch.float, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


class WindowedSDPAKernel(nn.Module):
    """Padded-sequence sliding-window attention kernel.

    Implements local attention via a pure-PyTorch additive mask — no external
    dependencies.  window_size controls the one-sided lookback (causal) or
    half-width (bidirectional).  Falls back to full SDPA when window_size >=
    the sequence length, preserving the AttnKernel interface throughout.
    """

    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.dropout = dropout

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        v: Float[Tensor, "b h s dh"],
        attn_mask: Float[Tensor, "b 1 n s"] | None,
        attn_bias: Float[Tensor, "h n s"] | None,
        is_causal: bool,
    ) -> Float[Tensor, "b h n dh"]:
        n, s = q.shape[-2], k.shape[-2]
        dropout_p = self.dropout if self.training else 0.0
        effective_causal = is_causal or self.causal

        if n <= self.window_size:
            # Window covers the full sequence — plain SDPA
            plain_mask: Tensor | None
            if attn_bias is not None:
                bias = attn_bias.unsqueeze(0)
                plain_mask = bias + (attn_mask if attn_mask is not None else 0.0)
            else:
                plain_mask = attn_mask
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=plain_mask,
                dropout_p=dropout_p,
                is_causal=effective_causal and plain_mask is None,
            )

        combined: Tensor = _build_window_mask(n, s, self.window_size, effective_causal, q.device)
        if attn_bias is not None:
            combined = combined + attn_bias.unsqueeze(0)
        if attn_mask is not None:
            combined = combined + attn_mask
        # is_causal=False because the window mask encodes the causal constraint
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined,
            dropout_p=dropout_p,
            is_causal=False,
        )


class VarlenWindowedSDPAKernel(nn.Module):
    """Packed variable-length sliding-window attention kernel.

    Primary path: torch.nn.attention.varlen.varlen_attn with a finite window
    parameter — requires CUDA and fp16 / bf16.

    Fallback: per-sequence loop with a PyTorch window mask — always available.

    window_size: one-sided lookback (causal) or half-width (bidirectional).
    """

    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.dropout = dropout
        # varlen_attn window_size convention: (left_tokens, right_tokens)
        self._varlen_window = (window_size, 0) if causal else (window_size // 2, window_size // 2)

    def forward(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        v: Float[Tensor, "nt h dh"],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Float[Tensor, "nt h dh"]:
        if _HAS_VARLEN_ATTN and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            cu = cu_seqlens.to(torch.int32)
            result = _varlen_attn(
                query=q,
                key=k,
                value=v,
                cu_seq_q=cu,
                cu_seq_k=cu,
                max_q=max_seqlen,
                max_k=max_seqlen,
                window_size=self._varlen_window,
            )
            assert isinstance(result, Tensor)
            return result

        # CPU / float32 fallback — loop with per-sequence window mask
        batch = cu_seqlens.shape[0] - 1
        outputs: list[Tensor] = []
        for i in range(batch):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            qi = rearrange(q[start:end], "n h d -> 1 h n d")
            ki = rearrange(k[start:end], "s h d -> 1 h s d")
            vi = rearrange(v[start:end], "s h d -> 1 h s d")
            n, s = qi.shape[-2], ki.shape[-2]
            window_mask = _build_window_mask(n, s, self.window_size, self.causal, q.device)
            dropout_p = self.dropout if self.training else 0.0
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=window_mask, dropout_p=dropout_p, is_causal=False
            )
            outputs.append(rearrange(out_i, "1 h n d -> n h d"))
        return torch.cat(outputs, dim=0)
