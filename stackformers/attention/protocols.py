from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from stackformers.sequence import PaddedInput, SequenceInput


@runtime_checkable
class AttnBias(Protocol):
    """Additive bias injected into attention logits before softmax.

    Receives the padded sequence input so implementations can use abs_positions.
    Returns a broadcastable bias (1|b, 1|h, n, s), or None if the bias is a no-op
    (which allows the packed path to use varlen_attn instead of SDPA).
    """

    def __call__(
        self,
        input: PaddedInput,
    ) -> Float[Tensor, "b h n s"] | None: ...


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps input → x."""

    def __call__(self, input: SequenceInput) -> Tensor: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Cross-attention from x to context: maps (x_input, ctx_input) → x."""

    def __call__(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor: ...
