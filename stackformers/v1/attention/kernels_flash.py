"""Optional FlashAttention varlen kernel.

Import flash-attn lazily inside __init__ — never at module load time.
Only available when flash-attn is installed (pip install flash-attn).
"""

from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class FlashVarlenKernel(nn.Module):
    """Packed-sequence attention via flash_attn_varlen_func.

    Requires: pip install flash-attn --no-build-isolation
    """

    def __init__(self, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = dropout

        try:
            from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]

            self._flash_fn = flash_attn_varlen_func
            self._available = True
        except ImportError as e:
            raise ImportError(
                "flash-attn is required for FlashVarlenKernel. "
                "Install with: pip install flash-attn --no-build-isolation"
            ) from e

    def forward(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        v: Float[Tensor, "nt h dh"],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Float[Tensor, "nt h dh"]:
        dropout_p = self.dropout if self.training else 0.0
        return self._flash_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
            causal=self.causal,
        )
