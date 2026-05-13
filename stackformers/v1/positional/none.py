from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class NoPosEncoding(nn.Module):
    """Null object for PosEncoding — passes q and k through unchanged."""

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        return q, k
