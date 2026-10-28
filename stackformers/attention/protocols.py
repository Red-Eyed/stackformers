from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from stackformers.attention.masking import CrossAttentionMask, SelfAttentionMask
    from stackformers.sequence import PackedTensor, PaddedTensor, SequenceInput


@runtime_checkable
class AttnBias(Protocol):
    """Additive bias injected into attention logits before softmax.

    Receives the padded sequence input so implementations can use abs_positions.
    Returns a broadcastable bias (1|b, 1|h, n, s), or None if the bias is a no-op
    (which allows the packed path to use varlen_attn instead of SDPA).
    """

    def __call__(
        self,
        input: PaddedTensor,
    ) -> Float[Tensor, "b h n s"] | None: ...


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps input → x."""

    def __call__(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> Tensor: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Map queries and context of the same layout to query-shaped embeddings."""

    @overload
    def __call__(
        self,
        x_input: PaddedTensor,
        ctx_input: PaddedTensor,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend between padded batches with the same batch size."""
        ...

    @overload
    def __call__(
        self,
        x_input: PackedTensor,
        ctx_input: PackedTensor,
        *,
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Attend between packed batches with the same document count."""
        ...
