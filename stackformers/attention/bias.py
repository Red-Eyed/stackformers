from __future__ import annotations

import math
import warnings
from typing import overload

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.sequence import PaddedInput, SequenceInput


class NoBiasBuilder(nn.Module):
    """Null object for AttnBiasBuilder — returns None (no additive bias)."""

    @overload
    def forward(self, q_input: SequenceInput, k_input: SequenceInput) -> None: ...
    @overload
    def forward(self, q_input: int, k_input: int, device: torch.device | None = None) -> None: ...

    def forward(  # type: ignore[misc]
        self,
        q_input: SequenceInput | int,
        k_input: SequenceInput | int,
        device: torch.device | None = None,
    ) -> None:
        return None


class ALiBiBuilder(nn.Module):
    """Attention with Linear Biases (Press et al., 2021).

    Adds a head-specific slope * |i - j| penalty to attention logits.
    Non-causal: bias[h, i, j] = -slope[h] * |i - j|
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

    @overload
    def forward(
        self, q_input: SequenceInput, k_input: SequenceInput
    ) -> Float[Tensor, "b h n s"] | None: ...
    @overload
    def forward(
        self, q_input: int, k_input: int, device: torch.device | None = None
    ) -> Float[Tensor, "h n s"] | None: ...

    def forward(  # type: ignore[misc]
        self,
        q_input: SequenceInput | int,
        k_input: SequenceInput | int,
        device: torch.device | None = None,
    ) -> Tensor | None:
        if isinstance(q_input, PaddedInput):
            assert isinstance(k_input, PaddedInput)
            q_pos = q_input.abs_positions
            k_pos = k_input.abs_positions
            dist = (q_pos[:, :, None] - k_pos[:, None, :]).abs().float()  # b n s
            slopes = self.slopes.to(device=q_pos.device)  # h
            return -(slopes[None, :, None, None] * dist[:, None, :, :])  # b h n s
        if isinstance(q_input, int):
            assert isinstance(k_input, int)
            n, s = q_input, k_input
            i_pos = torch.arange(n, device=device)
            j_pos = torch.arange(s, device=device)
            distance = (i_pos[:, None] - j_pos[None, :]).abs().float()  # n s
            slopes = self.slopes.to(device=device)  # h
            return -(slopes[:, None, None] * distance[None, :, :])  # h n s
        return None
