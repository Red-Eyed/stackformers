from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from stackformers.positional.config import RoPE1DConfig, YaRNConfig
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput


def _rotate_half(x: Tensor) -> Tensor:
    """Split x into [x1 | x2] halves and return [-x2 | x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_padded_unbatched(t: Tensor, freqs: Tensor) -> Tensor:
    """t: b h n dh, freqs: n dh — kept for rope2d compatibility."""
    cos = freqs.cos()
    sin = freqs.sin()
    return t * cos + _rotate_half(t) * sin


def _apply_rope_padded(t: Tensor, freqs: Tensor) -> Tensor:
    """t: b h n dh, freqs: b n dh"""
    cos = freqs.cos().unsqueeze(1)  # b 1 n dh
    sin = freqs.sin().unsqueeze(1)
    return t * cos + _rotate_half(t) * sin


def _apply_rope_packed(t: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE to packed tensor (nt h dh) with freqs (nt dh)."""
    cos = freqs.cos().unsqueeze(1)  # (nt, 1, dh) — broadcasts over h
    sin = freqs.sin().unsqueeze(1)
    return t * cos + _rotate_half(t) * sin


def _yarn_inv_freq(
    inv_freq: Tensor,
    cfg: YaRNConfig,
) -> Tensor:
    """Return YaRN-scaled inv_freq (Peng et al., 2023, NTK-by-parts)."""
    wavelen = 2.0 * math.pi / inv_freq
    low_wavelen = cfg.original_max_seq_len / cfg.beta_slow
    high_wavelen = cfg.original_max_seq_len / cfg.beta_fast
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

    Handles both padded (b h n dh) and packed (nt h dh) layouts via input type.
    Optionally accepts YaRNConfig for extended context via NTK-by-parts scaling.
    """

    def __init__(self, config: RoPE1DConfig) -> None:
        super().__init__()
        assert config.dim_head % 2 == 0, "dim_head must be even for RoPE"
        inv_freq = 1.0 / (
            config.base ** (torch.arange(0, config.dim_head, 2).float() / config.dim_head)
        )
        if config.yarn is not None:
            inv_freq = _yarn_inv_freq(inv_freq, config.yarn)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _freqs_from_padded_positions(self, positions: Tensor) -> Tensor:
        """positions: b n → freqs: b n dh"""
        pos = positions.to(dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("b n, d -> b n d", pos, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)

    @torch.no_grad()
    def _freqs_from_packed_positions(self, positions: Tensor) -> Tensor:
        """positions: nt → freqs: nt dh"""
        pos = positions.to(dtype=self.inv_freq.dtype)  # type: ignore[attr-defined]
        freqs = torch.einsum("n, d -> n d", pos, self.inv_freq)  # type: ignore[attr-defined]
        return torch.cat([freqs, freqs], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_input: SequenceInput,
        k_input: SequenceInput,
    ) -> tuple[Tensor, Tensor]:
        match q_input:
            case PaddedInput(abs_positions=q_pos):
                k_pos = k_input.abs_positions  # b s
                freqs_q = self._freqs_from_padded_positions(q_pos)
                freqs_k = self._freqs_from_padded_positions(k_pos)
                return _apply_rope_padded(q, freqs_q), _apply_rope_padded(k, freqs_k)
            case PackedInput(abs_positions=q_pos):
                k_pos = k_input.abs_positions  # nt
                freqs_q = self._freqs_from_packed_positions(q_pos)
                freqs_k = self._freqs_from_packed_positions(k_pos)
                return _apply_rope_packed(q, freqs_q), _apply_rope_packed(k, freqs_k)
