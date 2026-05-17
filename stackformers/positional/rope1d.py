from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.config import RoPE1DConfig, YaRNConfig


def _rotate_half(x: Tensor) -> Tensor:
    """Split x into [x1 | x2] halves and return [-x2 | x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_padded_unbatched(t: Tensor, freqs: Tensor) -> Tensor:
    """t: b h n dh, freqs: n dh — (n dh) broadcasts over (b h n dh) without unsqueeze."""
    cos = freqs.cos()
    sin = freqs.sin()
    return t * cos + _rotate_half(t) * sin


def _apply_rope(t: Tensor, freqs: Tensor) -> Tensor:
    """Works for padded (b h n dh)+(b n dh) and packed (nt h dh)+(nt dh).

    unsqueeze(1) inserts the head dim in both cases.
    Cast to t.dtype so float32 freqs don't upcast a float16 input.
    """
    cos = freqs.cos().to(dtype=t.dtype).unsqueeze(1)
    sin = freqs.sin().to(dtype=t.dtype).unsqueeze(1)
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
    def _freqs_from_positions(self, positions: Tensor) -> Tensor:
        """positions: (..., c) → freqs: (..., dh), uses first coordinate only.

        float32 cast ensures half-precision inputs don't lose precision in the outer product.
        """
        inv: Tensor = self.inv_freq  # type: ignore[assignment]
        pos = positions[..., 0].to(dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * inv.float()
        return torch.cat([freqs, freqs], dim=-1)

    def _encode(
        self,
        q: Tensor,
        k: Tensor,
        q_positions: Tensor,
        k_positions: Tensor,
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
