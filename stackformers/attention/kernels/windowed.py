from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stackformers.attention.kernels._mask import build_window_mask
from stackformers.attention.kernels.sdpa import _padding_mask
from stackformers.sequence import PaddedSequence, SequenceInfo


class WindowedSDPAKernel(nn.Module):
    """Padded-sequence sliding-window attention kernel.

    Implements local attention via a pure-PyTorch additive mask.
    Causal masking configured at construction.
    """

    def __init__(self, window_size: int, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.dropout = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_seq_info: SequenceInfo,
        k_seq_info: SequenceInfo | None,
        attn_bias: Tensor | None,
    ) -> Tensor:
        n, s = q.shape[-2], k.shape[-2]
        combined: Tensor = build_window_mask(n, s, self.window_size, self.causal, q.device)

        if attn_bias is not None:
            combined = combined + attn_bias
        if isinstance(k_seq_info, PaddedSequence):
            combined = combined + _padding_mask(k_seq_info.mask, q.dtype)

        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=combined, dropout_p=dropout_p, is_causal=False
        )
