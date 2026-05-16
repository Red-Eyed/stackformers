from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from stackformers.sequence import SequenceInfo, SequenceInput


@runtime_checkable
class AttnKernel(Protocol):
    """Compute scaled dot-product attention.

    Works for both padded (b h n dh) and packed (nt h dh) layouts —
    implementations dispatch on seq_info type.
    Causal masking is configured at construction, not passed per call.
    Implementations: SDPAKernel, WindowedSDPAKernel, VarlenSDPAKernel,
    VarlenWindowedSDPAKernel.
    """

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_seq_info: SequenceInfo,
        k_seq_info: SequenceInfo | None,
    ) -> Tensor: ...


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps input → x.

    Implementation: SelfAttention.
    """

    def __call__(
        self,
        input: SequenceInput,
    ) -> Tensor: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Cross-attention from x to context: maps (x_input, ctx_input) → x.

    Implementation: CrossAttention.
    """

    def __call__(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput,
    ) -> Tensor: ...
