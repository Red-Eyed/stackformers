from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from stackformers.attention.kernels._mask import build_window_mask

try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    _HAS_VARLEN_ATTN = True
except ImportError:
    _HAS_VARLEN_ATTN = False


class VarlenWindowedSDPAKernel(nn.Module):
    """Packed variable-length sliding-window attention kernel.

    Primary path: torch.nn.attention.varlen.varlen_attn with a finite window
    parameter — requires CUDA and fp16 / bf16.

    Fallback: per-sequence loop with a PyTorch window mask.

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

        import warnings

        if not _HAS_VARLEN_ATTN:
            reason = "torch.nn.attention.varlen.varlen_attn is unavailable (requires PyTorch ≥ 2.5)"
        elif not q.is_cuda:
            reason = f"tensor is on {q.device}, not CUDA"
        else:
            reason = f"dtype is {q.dtype}, not float16 or bfloat16"
        warnings.warn(
            f"VarlenWindowedSDPAKernel falling back to slow per-sequence loop: {reason}. "
            "This is O(batch) kernel launches instead of one fused call.",
            UserWarning,
            stacklevel=2,
        )

        batch = cu_seqlens.shape[0] - 1
        outputs: list[Tensor] = []
        for i in range(batch):
            start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
            qi = rearrange(q[start:end], "n h d -> 1 h n d")
            ki = rearrange(k[start:end], "s h d -> 1 h s d")
            vi = rearrange(v[start:end], "s h d -> 1 h s d")
            n, s = qi.shape[-2], ki.shape[-2]
            window_mask = build_window_mask(n, s, self.window_size, self.causal, q.device)
            dropout_p = self.dropout if self.training else 0.0
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=window_mask, dropout_p=dropout_p, is_causal=False
            )
            outputs.append(rearrange(out_i, "1 h n d -> n h d"))
        return torch.cat(outputs, dim=0)
