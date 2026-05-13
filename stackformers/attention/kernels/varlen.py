from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from stackformers.sequence import PackedSequence, SequenceInfo

try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    _HAS_VARLEN_ATTN = True
except ImportError:
    _HAS_VARLEN_ATTN = False


class VarlenSDPAKernel(nn.Module):
    """Packed variable-length SDPA kernel (full attention, no windowing).

    Primary path: torch.nn.attention.varlen.varlen_attn — requires CUDA + fp16/bf16.
    Fallback: per-sequence loop over F.scaled_dot_product_attention.
    Causal masking configured at construction.
    """

    def __init__(self, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = dropout
        self._window_size = (-1, 0) if causal else (-1, -1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_seq_info: SequenceInfo,
        k_seq_info: SequenceInfo | None,
        _attn_bias: Tensor | None,
    ) -> Tensor:
        assert isinstance(q_seq_info, PackedSequence), "VarlenSDPAKernel requires PackedSequence"
        k_info = k_seq_info if isinstance(k_seq_info, PackedSequence) else q_seq_info

        if _HAS_VARLEN_ATTN and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            result = _varlen_attn(
                query=q,
                key=k,
                value=v,
                cu_seq_q=q_seq_info.cu_seqlens.to(torch.int32),
                cu_seq_k=k_info.cu_seqlens.to(torch.int32),
                max_q=q_seq_info.max_seqlen,
                max_k=k_info.max_seqlen,
                window_size=self._window_size,
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
            f"VarlenSDPAKernel falling back to slow per-sequence loop: {reason}.",
            UserWarning,
            stacklevel=2,
        )

        batch = q_seq_info.cu_seqlens.shape[0] - 1
        outputs: list[Tensor] = []
        for i in range(batch):
            qs = int(q_seq_info.cu_seqlens[i].item())
            qe = int(q_seq_info.cu_seqlens[i + 1].item())
            ks = int(k_info.cu_seqlens[i].item())
            ke = int(k_info.cu_seqlens[i + 1].item())
            qi = rearrange(q[qs:qe], "n h d -> 1 h n d")
            ki = rearrange(k[ks:ke], "s h d -> 1 h s d")
            vi = rearrange(v[ks:ke], "s h d -> 1 h s d")
            dropout_p = self.dropout if self.training else 0.0
            out_i = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, is_causal=self.causal)
            outputs.append(rearrange(out_i, "1 h n d -> n h d"))
        return torch.cat(outputs, dim=0)
