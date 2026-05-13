from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class NoBiasBuilder(nn.Module):
    """Null object for AttnBiasBuilder — returns None (no additive bias)."""

    def forward(
        self,
        n: int,
        s: int,
        device: torch.device,
    ) -> None:
        return None


class ALiBiBuilder(nn.Module):
    """Attention with Linear Biases (Press et al., 2021).

    Adds a head-specific slope * |i - j| penalty to attention logits.
    Non-causal: bias[h, i, j] = -slope[h] * |i - j|
    Causal: bias[h, i, j] = -slope[h] * (i - j) for j <= i, else -inf (handled by kernel)
    """

    slopes: Tensor  # populated by register_buffer

    def __init__(self, heads: int, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        if not math.log2(heads).is_integer():
            warnings.warn(
                f"ALiBiBuilder: heads={heads} is not a power of 2. "
                "Slopes are interpolated from the nearest power-of-2 formula, "
                "which deviates from the original ALiBi paper. "
                "Consider using heads=2^k (e.g. "
                f"{2 ** math.floor(math.log2(heads))} or {2 ** math.ceil(math.log2(heads))}).",
                UserWarning,
                stacklevel=2,
            )
        slopes = self._get_slopes(heads)
        self.register_buffer("slopes", slopes, persistent=True)

    @staticmethod
    def _get_slopes(heads: int) -> Float[Tensor, "h"]:
        def _slopes_for_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

        if math.log2(heads).is_integer():
            return torch.tensor(_slopes_for_power_of_2(heads))

        # For non-powers-of-2: take next power of 2 slopes + interleave half
        closest_pow2 = 2 ** math.floor(math.log2(heads))
        base = _slopes_for_power_of_2(closest_pow2)
        extra = _slopes_for_power_of_2(2 * closest_pow2)[0::2]
        slopes = base + extra[: heads - closest_pow2]
        return torch.tensor(slopes)

    def forward(
        self,
        n: int,
        s: int,
        device: torch.device,
    ) -> Float[Tensor, "h n s"]:
        i_pos = torch.arange(n, device=device)
        j_pos = torch.arange(s, device=device)
        distance = (i_pos[:, None] - j_pos[None, :]).abs().float()  # (n, s)

        slopes = self.slopes.to(device=device)  # (h,)
        bias = -slopes[:, None, None] * distance[None, :, :]  # (h, n, s)
        return bias
