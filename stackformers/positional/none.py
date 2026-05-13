from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class NoPosEncoding(nn.Module):
    """Null object for PosEncoding and PackedPosEncoding — passes q, k unchanged."""

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        return q, k

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        _position_ids: Int[Tensor, "nt"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]:
        return q, k
