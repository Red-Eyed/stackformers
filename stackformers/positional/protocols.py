from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float, Int
from torch import Tensor


@runtime_checkable
class PosEncoding(Protocol):
    """Apply positional encoding to padded (b, h, n, dh) query and key tensors.

    Null implementation: NoPosEncoding.
    """

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]: ...


@runtime_checkable
class PackedPosEncoding(Protocol):
    """Apply positional encoding to packed (nt, h, dh) query and key tensors.

    Position ids locate each token within its document.
    Null implementation: NoPosEncoding.
    """

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        position_ids: Int[Tensor, "nt"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]: ...
