from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.config import RoPE2DConfig
from stackformers.positional.rope1d import (
    _apply_rope_packed,
)
from stackformers.positional.rope1d import (
    _apply_rope_padded_unbatched as _apply_rope,
)
from stackformers.sequence import PackedInput, SequenceInput


class RotaryEmbedding2D(nn.Module):
    """2-D Rotary Position Embedding for (row, col) positions.

    Conforms to PosEncoding: abs_positions must have c=2 (row, col per token).
    Splits dim_head evenly: first half encodes row, second half column.

    Padded path: positions are shared across the batch (standard grid layout).
    Packed path: each token carries its own (row, col) in abs_positions.
    """

    def __init__(self, config: RoPE2DConfig) -> None:
        super().__init__()
        assert config.dim_head % 4 == 0, "dim_head must be divisible by 4 for 2-D RoPE"
        half_dh = config.dim_head // 2
        inv_freq = 1.0 / (config.base ** (torch.arange(0, half_dh, 2).float() / half_dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.half_dh = half_dh

    @torch.no_grad()
    def _build_freqs(
        self, ids: Float[Tensor, "n"], device: torch.device
    ) -> Float[Tensor, "n dh_half"]:
        ids = ids.to(device=device, dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", ids, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        q_input: SequenceInput,
        k_input: SequenceInput,
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        match q_input:
            case PackedInput():
                # Per-token 2D positions: abs_positions is (nt, 2)
                q_pos = q_input.abs_positions  # nt 2
                k_pos = k_input.abs_positions  # nt 2
                freqs_q = torch.cat(
                    [
                        self._build_freqs(q_pos[:, 0], q.device),
                        self._build_freqs(q_pos[:, 1], q.device),
                    ],
                    dim=-1,
                )
                freqs_k = torch.cat(
                    [
                        self._build_freqs(k_pos[:, 0], k.device),
                        self._build_freqs(k_pos[:, 1], k.device),
                    ],
                    dim=-1,
                )
                return _apply_rope_packed(q, freqs_q), _apply_rope_packed(k, freqs_k)
            case _:
                # Padded: positions shared across batch (first item), abs_positions is (b, n, 2)
                q_pos = q_input.abs_positions[0]  # n 2
                k_pos = k_input.abs_positions[0]  # s 2
                freqs_q = torch.cat(
                    [
                        self._build_freqs(q_pos[:, 0], q.device),
                        self._build_freqs(q_pos[:, 1], q.device),
                    ],
                    dim=-1,
                )
                freqs_k = torch.cat(
                    [
                        self._build_freqs(k_pos[:, 0], k.device),
                        self._build_freqs(k_pos[:, 1], k.device),
                    ],
                    dim=-1,
                )
                return _apply_rope(q, freqs_q), _apply_rope(k, freqs_k)
