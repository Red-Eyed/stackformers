"""Validate dynamic layout pairs before calling a layout-specific attention contract."""

from torch import Tensor

from stackformers.attention.protocols import CrossAttn
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput


def call_cross_attention(
    attention: CrossAttn, queries: SequenceInput, context: SequenceInput
) -> Tensor:
    """Narrow matching layouts together, or reject a mixed pair with ValueError."""
    match queries, context:
        case PaddedInput(), PaddedInput():
            return attention(queries, context)
        case PackedInput(), PackedInput():
            return attention(queries, context)
        case _:
            raise ValueError("cross-attention inputs must have matching layouts")
