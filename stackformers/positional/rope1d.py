from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from stackformers.positional.config import YaRNConfig


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


def _yarn_inv_freq(
    inv_freq: Float[Tensor, "dh_half"],
    cfg: YaRNConfig,
) -> Float[Tensor, "dh_half"]:
    """Return YaRN-scaled inv_freq (Peng et al., 2023, NTK-by-parts).

    Wavelength thresholds split dimensions into three bands:
      - high-freq (short wavelength): unchanged
      - low-freq  (long  wavelength): scaled down by cfg.scale
      - middle: smooth linear blend controlled by cfg.beta_fast / beta_slow
    All ops are on tensors; this function is safe to call from __init__.
    """
    wavelen = 2.0 * math.pi / inv_freq  # (dh_half,)
    low_wavelen = cfg.original_max_seq_len / cfg.beta_slow
    high_wavelen = cfg.original_max_seq_len / cfg.beta_fast

    # smooth ∈ [0, 1]: 0 at low-freq boundary, 1 at high-freq boundary
    smooth = (
        (cfg.original_max_seq_len / wavelen - cfg.beta_slow) / (cfg.beta_fast - cfg.beta_slow)
    ).clamp(0.0, 1.0)

    scaled = inv_freq / cfg.scale
    blended = smooth * inv_freq + (1.0 - smooth) * scaled
    return torch.where(
        wavelen < high_wavelen, inv_freq, torch.where(wavelen > low_wavelen, scaled, blended)
    )


class RotaryEmbedding1D(nn.Module):
    """1-D Rotary Position Embedding (Su et al., 2021).

    Optionally accepts a YaRNConfig to extend the effective context length
    via NTK-by-parts frequency scaling (Peng et al., 2023).  The scaling
    modifies inv_freq once at construction — forward() is unchanged.

    Math ref: x-transformers RotaryEmbedding / apply_rotary_pos_emb.
    """

    def __init__(self, dim_head: int, base: int = 10_000, yarn: YaRNConfig | None = None) -> None:
        super().__init__()
        assert dim_head % 2 == 0, "dim_head must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        if yarn is not None:
            inv_freq = _yarn_inv_freq(inv_freq, yarn)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _build_freqs(self, seq_len: int, device: torch.device) -> Float[Tensor, "n dh"]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", positions, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)  # (n, dh)

    @torch.no_grad()
    def _build_freqs_from_ids(
        self, position_ids: Int[Tensor, "nt"], device: torch.device
    ) -> Float[Tensor, "nt dh"]:
        ids = position_ids.to(device=device, dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", ids, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)  # (nt, dh)

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        # Always build both frequency tensors — no branch on n vs s so that
        # torch.export can trace with dynamic sequence lengths.
        freqs_q = self._build_freqs(q.shape[-2], q.device)
        freqs_k = self._build_freqs(k.shape[-2], k.device)
        return _apply_rope(q, freqs_q), _apply_rope(k, freqs_k)

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        position_ids: Int[Tensor, "nt"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]:
        """Apply RoPE to packed (varlen) head tensors using per-token position ids."""
        freqs = self._build_freqs_from_ids(position_ids, q.device)  # (nt, dh)
        cos = freqs.cos().unsqueeze(1)  # (nt, 1, dh) — broadcasts over h
        sin = freqs.sin().unsqueeze(1)
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out, k_out
