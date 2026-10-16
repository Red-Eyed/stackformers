"""Validate dynamic layout pairs before calling a layout-specific attention contract."""

from torch import Tensor

from stackformers._result import unwrap_or_raise
from stackformers.attention.masking import CrossAttentionMask, SelfAttentionMask
from stackformers.attention.protocols import CrossAttn, SelfAttn
from stackformers.result import Err, Ok, Result
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput


def matching_layouts(
    queries: SequenceInput, context: SequenceInput
) -> Result[tuple[PaddedInput, PaddedInput] | tuple[PackedInput, PackedInput], ValueError]:
    """Return a jointly narrowed pair, carrying a layout mismatch as failure data."""
    match queries, context:
        case PaddedInput(), PaddedInput():
            return Ok((queries, context))
        case PackedInput(), PackedInput():
            return Ok((queries, context))
        case _:
            return Err(ValueError("cross-attention inputs must have matching layouts"))


def call_cross_attention(
    attention: CrossAttn,
    queries: SequenceInput,
    context: SequenceInput,
    *,
    cross_attn_mask: CrossAttentionMask = None,
) -> Tensor:
    """Adapt the layout result to the established tensor/ValueError call contract."""
    match unwrap_or_raise(matching_layouts(queries, context)):
        case PaddedInput() as padded_queries, PaddedInput() as padded_context:
            if cross_attn_mask is None:
                return attention(padded_queries, padded_context)
            return attention(padded_queries, padded_context, cross_attn_mask=cross_attn_mask)
        case PackedInput() as packed_queries, PackedInput() as packed_context:
            if cross_attn_mask is None:
                return attention(packed_queries, packed_context)
            return attention(packed_queries, packed_context, cross_attn_mask=cross_attn_mask)


def call_self_attention(
    attention: SelfAttn, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
) -> Tensor:
    """Preserve legacy collaborator calls when no visibility override was requested."""
    match self_attn_mask:
        case Tensor():
            return attention(input, self_attn_mask=self_attn_mask)
        case "default":
            return attention(input)
        case _:
            return attention(input, self_attn_mask=self_attn_mask)
