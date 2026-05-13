from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
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
