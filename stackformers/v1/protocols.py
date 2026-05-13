from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class PosEncoding(Protocol):
    """Apply positional encoding to query and key tensors.

    Null implementation: NoPosEncoding (returns q, k unchanged).
    """

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]: ...


@runtime_checkable
class AttnBiasBuilder(Protocol):
    """Build an additive attention bias matrix.

    Returns None when no bias should be added (null object: NoBiasBuilder).
    """

    def forward(
        self,
        n: int,
        s: int,
        device: torch.device,
    ) -> Float[Tensor, "h n s"] | None: ...


@runtime_checkable
class AttnKernel(Protocol):
    """Compute scaled dot-product attention given projected heads.

    Implementations: SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel.
    """

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        v: Float[Tensor, "b h s dh"],
        attn_mask: Float[Tensor, "b 1 n s"] | None,
        attn_bias: Float[Tensor, "h n s"] | None,
        is_causal: bool,
    ) -> Float[Tensor, "b h n dh"]: ...


@runtime_checkable
class Norm(Protocol):
    """Apply layer normalisation."""

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]: ...
