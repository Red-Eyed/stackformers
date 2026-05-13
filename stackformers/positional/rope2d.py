from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.rope1d import _apply_rope


class RotaryEmbedding2D(nn.Module):
    """2-D Rotary Position Embedding for (height, width) grids.

    Splits dim_head evenly: first half encodes row position, second half column.
    Grid positions passed as (row_ids, col_ids) tensors of shape (n,).
    """

    def __init__(self, dim_head: int, base: int = 10_000) -> None:
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2-D RoPE"
        half_dh = dim_head // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dh, 2).float() / half_dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.half_dh = half_dh

    @torch.no_grad()
    def _build_freqs(
        self,
        ids: Float[Tensor, "n"],
        device: torch.device,
    ) -> Float[Tensor, "n dh_half"]:
        ids = ids.to(device=device, dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", ids, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)  # (n, half_dh)

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        row_ids: Float[Tensor, "n"],
        col_ids: Float[Tensor, "n"],
        kv_row_ids: Float[Tensor, "s"] | None = None,
        kv_col_ids: Float[Tensor, "s"] | None = None,
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        device = q.device
        kv_row_ids = kv_row_ids if kv_row_ids is not None else row_ids
        kv_col_ids = kv_col_ids if kv_col_ids is not None else col_ids

        freqs_row_q = self._build_freqs(row_ids, device)
        freqs_col_q = self._build_freqs(col_ids, device)
        freqs_row_k = self._build_freqs(kv_row_ids, device)
        freqs_col_k = self._build_freqs(kv_col_ids, device)

        freqs_q = torch.cat([freqs_row_q, freqs_col_q], dim=-1)  # (n, dh)
        freqs_k = torch.cat([freqs_row_k, freqs_col_k], dim=-1)

        q_out = _apply_rope(q, freqs_q)
        k_out = _apply_rope(k, freqs_k)
        return q_out, k_out
