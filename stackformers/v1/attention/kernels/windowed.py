from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from stackformers.v1.attention.kernels._mask import build_window_mask


class WindowedSDPAKernel(nn.Module):
    """Padded-sequence sliding-window attention kernel.

    Implements local attention via a pure-PyTorch additive mask — no external
    dependencies. window_size controls the one-sided lookback (causal) or
    half-width (bidirectional). Falls back to full SDPA when window_size >=
    the sequence length.
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
            plain_mask: Tensor | None
            if attn_bias is not None:
                plain_mask = attn_bias.unsqueeze(0) + (attn_mask if attn_mask is not None else 0.0)
            else:
                plain_mask = attn_mask
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=plain_mask,
                dropout_p=dropout_p,
                is_causal=effective_causal and plain_mask is None,
            )

        combined: Tensor = build_window_mask(n, s, self.window_size, effective_causal, q.device)
        if attn_bias is not None:
            combined = combined + attn_bias.unsqueeze(0)
        if attn_mask is not None:
            combined = combined + attn_mask
        return F.scaled_dot_product_attention(q, k, v, attn_mask=combined, dropout_p=dropout_p, is_causal=False)
