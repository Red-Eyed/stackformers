from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

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
    available on CPU and float32.
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

        batch = cu_seqlens.shape[0] - 1
        outputs: list[Tensor] = []
        for i in range(batch):
            start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
            qi = rearrange(q[start:end], "n h d -> 1 h n d")
            ki = rearrange(k[start:end], "s h d -> 1 h s d")
            vi = rearrange(v[start:end], "s h d -> 1 h s d")
            dropout_p = self.dropout if self.training else 0.0
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=self.causal
            )
            outputs.append(rearrange(out_i, "1 h n d -> n h d"))
        return torch.cat(outputs, dim=0)
