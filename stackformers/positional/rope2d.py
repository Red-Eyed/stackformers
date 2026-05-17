from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.config import RoPE2DConfig
from stackformers.positional.rope1d import _apply_rope


class RotaryEmbedding2D(nn.Module):
    """2-D Rotary Position Embedding for (row, col) positions.

    Conforms to PosEncoding: positions must have c=2 (row, col per token).
    Splits dim_head evenly: first half encodes row, second half column.
    """

    def __init__(self, config: RoPE2DConfig) -> None:
        super().__init__()
        assert config.dim_head % 4 == 0, "dim_head must be divisible by 4 for 2-D RoPE"
        half_dh = config.dim_head // 2
        inv_freq = 1.0 / (config.base ** (torch.arange(0, half_dh, 2).float() / half_dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _freqs_from_positions(self, positions: Tensor) -> Tensor:
        """positions: (..., 2) → freqs: (..., dh)"""
        inv: Tensor = self.inv_freq  # type: ignore[assignment]
        row = positions[..., 0].to(dtype=inv.dtype).unsqueeze(-1) * inv  # (..., dh//4)
        col = positions[..., 1].to(dtype=inv.dtype).unsqueeze(-1) * inv
        half = lambda f: torch.cat([f, f], dim=-1)  # noqa: E731
        return torch.cat([half(row), half(col)], dim=-1)  # (..., dh)

    def _encode(
        self, q: Tensor, k: Tensor, q_positions: Tensor, k_positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        return (
            _apply_rope(q, self._freqs_from_positions(q_positions)),
            _apply_rope(k, self._freqs_from_positions(k_positions)),
        )

    def forward_padded(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        q_positions: Float[Tensor, "b n c"],
        k_positions: Float[Tensor, "b s c"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        return self._encode(q, k, q_positions, k_positions)

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        q_positions: Float[Tensor, "nt c"],
        k_positions: Float[Tensor, "nt c"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]:
        return self._encode(q, k, q_positions, k_positions)
