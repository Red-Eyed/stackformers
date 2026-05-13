from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class Norm(Protocol):
    """Apply layer normalisation to (b, n, d) tensors.

    Implementation: RMSNorm.
    """

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]: ...
