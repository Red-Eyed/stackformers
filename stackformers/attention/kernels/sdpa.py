from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stackformers.sequence import PaddedSequence, SequenceInfo


def _padding_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert bool key-padding mask (True=valid) to additive float bias (b 1 1 s)."""
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias.masked_fill_(~mask, torch.finfo(dtype).min)
    return bias.view(mask.shape[0], 1, 1, mask.shape[1])


class SDPAKernel(nn.Module):
    """Padded-sequence SDPA kernel. Causal masking configured at construction."""

    def __init__(self, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
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
        attn_mask: Tensor | None = None
        if isinstance(k_seq_info, PaddedSequence):
            attn_mask = _padding_mask(k_seq_info.mask, q.dtype)

        if attn_bias is not None:
            combined = attn_bias + (attn_mask if attn_mask is not None else 0.0)
        else:
            combined = attn_mask

        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined,
            dropout_p=dropout_p,
            is_causal=self.causal and combined is None,
        )
