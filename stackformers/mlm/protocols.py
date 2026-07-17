from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Bool, Float
from torch import Tensor

from stackformers.sequence import SequenceInput


@runtime_checkable
class EncoderLike(Protocol):
    """Any encoder mapping a sequence to per-token embeddings of the same layout."""

    def __call__(self, input: SequenceInput) -> Tensor: ...


@runtime_checkable
class MaskingStrategy(Protocol):
    """Select which token positions to corrupt for reconstruction.

    Returns True at positions to mask, shaped like the leading (non-feature) dims of
    input.x — (b, n) for PaddedInput, (nt,) for PackedInput. Implementations dispatch
    on the SequenceInput variant internally, so callers never need to know the layout.
    """

    def __call__(self, input: SequenceInput) -> Bool[Tensor, "*batch"]: ...


@runtime_checkable
class ReconstructionHead(Protocol):
    """Predict masked tokens from encoder output and score against the clean target.

    Both arguments are already gathered down to just the masked positions (m = number
    of masked tokens across the batch or pack). Returns a scalar loss.
    """

    def __call__(
        self,
        encoder_output_at_masked: Float[Tensor, "m d"],
        target_at_masked: Float[Tensor, "m d"],
    ) -> Float[Tensor, ""]: ...
