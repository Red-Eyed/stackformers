"""Validate dynamic layout pairs before calling a layout-specific attention contract."""

from returns.result import Failure, Result, Success
from torch import Tensor

from stackformers._result import unwrap_or_raise
from stackformers.attention.protocols import CrossAttn
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput


def matching_layouts(
    queries: SequenceInput, context: SequenceInput
) -> Result[tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput], ValueError]:
    """Return a jointly narrowed pair, carrying a layout mismatch as failure data."""
    match queries, context:
        case PaddedInput(), PaddedInput():
            return Success((queries, context))
        case PackedInput(), PackedInput():
            return Success((queries, context))
        case _:
            return Failure(ValueError("cross-attention inputs must have matching layouts"))


def call_cross_attention(
    attention: CrossAttn, queries: SequenceInput, context: SequenceInput
) -> Tensor:
    """Adapt the layout result to the established tensor/ValueError call contract."""
    match unwrap_or_raise(matching_layouts(queries, context)):
        case PaddedInput() as padded_queries, PaddedInput() as padded_context:
            return attention(padded_queries, padded_context)
        case PackedInput() as packed_queries, PackedInput() as packed_context:
            return attention(packed_queries, packed_context)
