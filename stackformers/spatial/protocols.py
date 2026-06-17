from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from stackformers.spatial.input import SpatialInput


@runtime_checkable
class SpatialAttn(Protocol):
    """Attention over a 2-D grid of tokens: maps SpatialInput → x (b, n, d).

    Implementations: WindowAttention2D, SpatialReductionAttention.
    """

    def __call__(self, input: SpatialInput) -> Float[Tensor, "b n d"]: ...


@runtime_checkable
class KVReduction(Protocol):
    """Spatially downsample the key/value context before attention.

    Returns the reduced context tokens and their grid positions, so the caller can
    apply position encoding to the coarsened keys. Null implementation: NoKVReduction.
    """

    def __call__(
        self,
        x: Float[Tensor, "b n d"],
        grid: tuple[int, int],
    ) -> tuple[Float[Tensor, "b s d"], Float[Tensor, "b s c"]]: ...
