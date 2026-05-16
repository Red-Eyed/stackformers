from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from stackformers.sequence import SequenceInput


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps input → x.

    Implementation: SelfAttention.
    """

    def __call__(self, input: SequenceInput) -> Tensor: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Cross-attention from x to context: maps (x_input, ctx_input) → x.

    Implementation: CrossAttention.
    """

    def __call__(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor: ...
