from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def _rotate_half(x: Float[Tensor, "b h n dh"]) -> Float[Tensor, "b h n dh"]:
    """Split x into [x1 | x2] halves and return [-x2 | x1].

    Matches the 'halved' RoPE convention where freqs = cat([θ, θ]).
    Pairs (x[i], x[i+d/2]) both rotate by the same angle θ_i.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    t: Float[Tensor, "b h n dh"],
    freqs: Float[Tensor, "n dh"],
) -> Float[Tensor, "b h n dh"]:
    """Apply RoPE frequencies to tensor t."""
    cos = freqs.cos()
    sin = freqs.sin()
    return t * cos + _rotate_half(t) * sin


class RotaryEmbedding1D(nn.Module):
    """1-D Rotary Position Embedding (Su et al., 2021).

    Math ref: x-transformers RotaryEmbedding / apply_rotary_pos_emb.
    Rewritten using einops; no view/reshape.
    """

    def __init__(self, dim_head: int, base: int = 10_000) -> None:
        super().__init__()
        assert dim_head % 2 == 0, "dim_head must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _build_freqs(self, seq_len: int, device: torch.device) -> Float[Tensor, "n dh"]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", positions, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)  # (n, dh)

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        n = q.shape[-2]
        s = k.shape[-2]
        device = q.device

        freqs_q = self._build_freqs(n, device)
        freqs_k = self._build_freqs(s, device) if s != n else freqs_q

        q_out = _apply_rope(q, freqs_q)
        k_out = _apply_rope(k, freqs_k)
        return q_out, k_out
