from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from jaxtyping import Float

    from stackformers.positional.config import NoPosEncodingConfig


class NoPosEncoding(nn.Module):
    """Null object for PosEncoding — passes q, k unchanged regardless of layout."""

    def __init__(self, _config: NoPosEncodingConfig | None = None) -> None:
        """Construct a parameter-free identity encoding."""
        super().__init__()

    def forward_padded(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        q_positions: Float[Tensor, "b n c"],
        k_positions: Float[Tensor, "b s c"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        return q, k

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        q_positions: Float[Tensor, "nt c"],
        k_positions: Float[Tensor, "nt c"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]:
        return q, k
