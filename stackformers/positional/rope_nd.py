from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.config import RoPENDConfig
from stackformers.positional.rope1d import _apply_rope


def _frequency_ladder(config: RoPENDConfig) -> Tensor:
    """Band frequencies in *coordinate units*, from finest resolvable to whole-domain.

    Deliberately has no ``base``. RoPE's usual ``base ** (-2i/d)`` ladder pins its fastest band
    at ω=1, i.e. a wavelength of exactly 2π coordinate units, whatever the base — so ``base``
    can only stretch the slow end, and the ladder lands correctly only when tokens happen to
    sit one unit apart. That holds for text and for patch grids; it is meaningless for
    continuous coordinates, where the useful band range is fixed by the data instead:

        ω_hi = π / r_min                  → shortest wavelength = 2·r_min  (Nyquist on the
                                            finest separation the model must resolve)
        ω_lo = 2π / (headroom · r_max)    → longest wavelength spans the whole domain

    So the ladder is a function of the *dynamic range* r_max / r_min alone, and is invariant
    to the units the coordinates are expressed in — metres, pixels, or normalised to [0, 1]
    all give the same encoding. That is precisely the property ``base`` does not have.
    """
    omega_hi = math.pi / config.r_min
    omega_lo = 2.0 * math.pi / (config.headroom * config.r_max)
    steps = torch.linspace(math.log(omega_hi), math.log(omega_lo), config.bands_per_axis)
    return torch.exp(steps)  # fastest band first, matching the 1-D/2-D convention


class RotaryEmbeddingND(nn.Module):
    """Rotary position embedding over c spatial dimensions, for continuous coordinates.

    Generalises RotaryEmbedding1D (c=1) and RotaryEmbedding2D (c=2) to any c: ``dim_head`` is
    split into c equal blocks, one per axis, and each block is rotated by that axis'
    coordinate. Attention therefore depends only on the relative offset p_i − p_j — exactly,
    and in every dimension — so it is translation invariant by construction, with no centring
    step and no bias tensor. It is *not* rotation invariant: it encodes direction as well as
    distance. Where the global frame is arbitrary, train with rotation augmentation; buying
    exact rotation invariance in the architecture costs either an O(n²) attention bias or a
    discontinuous canonicalisation of the input frame.

    Unlike RoPE1DConfig/RoPE2DConfig this takes no ``base`` — see :func:`_frequency_ladder`.
    """

    def __init__(self, config: RoPENDConfig) -> None:
        super().__init__()
        self.coords = config.coords
        self.register_buffer("inv_freq", _frequency_ladder(config), persistent=False)

    @torch.no_grad()
    def _freqs_from_positions(self, positions: Tensor) -> Tensor:
        """positions: (..., c) → freqs: (..., dh), laid out as [axis_0 | … | axis_c-1] twice.

        _rotate_half pairs channel i with channel i + dh/2, so both members of a pair must
        carry the same angle for the map to be a rotation. Duplicating the *concatenated*
        per-axis block satisfies that; duplicating each axis block in place would pair one
        axis' angle against another's and yield a squeeze rather than a rotation — not
        orthogonal, and not a function of relative position.

        float32 keeps half-precision coordinates from losing the outer product's precision.
        """
        inv: Tensor = self.inv_freq  # type: ignore[assignment]
        pos = positions.to(dtype=torch.float32)
        per_axis = [pos[..., i].unsqueeze(-1) * inv.float() for i in range(self.coords)]
        half = torch.cat(per_axis, dim=-1)  # (..., dh//2)
        return torch.cat([half, half], dim=-1)  # (..., dh)

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
