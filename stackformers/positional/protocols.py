from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from stackformers.sequence import SequenceInput


@runtime_checkable
class PosEncoding(Protocol):
    """Apply positional encoding to query and key tensors.

    Works for both padded (b h n dh) and packed (nt h dh) layouts —
    implementations dispatch on input type.
    Null implementation: NoPosEncoding.
    """

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_input: SequenceInput,
        k_input: SequenceInput,
    ) -> tuple[Tensor, Tensor]: ...
