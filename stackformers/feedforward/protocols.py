from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


@runtime_checkable
class FeedForward(Protocol):
    """Apply a feed-forward transformation to (b, n, d) token embeddings."""

    def __call__(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]: ...
