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


class VarlenSDPAKernel(nn.Module):
    """Packed-sequence SDPA kernel.

    Iterates over sequences in the pack and runs SDPA per sequence, then
    stacks into a flat output.  Import flash-attn inside forward for optional
    fast path (never at module load time).
    """

    def __init__(self, dropout: float = 0.0, causal: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.causal = causal

    def forward(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        v: Float[Tensor, "nt h dh"],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Float[Tensor, "nt h dh"]:
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


class WindowedSDPAKernel(nn.Module):
    """Local sliding-window attention kernel.

    Uses the `local-attention` library (imported lazily inside __init__).
    Falls back to full SDPA when window_size >= sequence length.
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

        try:
            from local_attention import LocalAttention

            self._local_attn = LocalAttention(
                window_size=window_size,
                causal=causal,
                autopad=True,
                dropout=dropout,
            )
            self._use_local = True
        except ImportError:
            self._use_local = False

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        v: Float[Tensor, "b h s dh"],
        attn_mask: Float[Tensor, "b 1 n s"] | None,
        attn_bias: Float[Tensor, "h n s"] | None,
        is_causal: bool,
    ) -> Float[Tensor, "b h n dh"]:
        n = q.shape[-2]

        if self._use_local and n > self.window_size:
            return self._local_attn(q, k, v)  # type: ignore[no-any-return]

        # Fallback: full SDPA
        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
