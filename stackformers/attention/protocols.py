from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from stackformers.sequence import SequenceInfo


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
        attn_bias: Tensor | None,
    ) -> Tensor: ...


@runtime_checkable
class AttnBiasBuilder(Protocol):
    """Build an additive attention bias matrix.

    Returns None when no bias should be added.
    Null implementation: NoBiasBuilder.
    """

    def forward(
        self,
        n: int,
        s: int,
        device,
    ) -> Tensor | None: ...


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps (x, seq_info) → x.

    Implementation: SelfAttention.
    """

    def __call__(
        self,
        x: Tensor,
        seq_info: SequenceInfo,
    ) -> Tensor: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Cross-attention from x to context: maps (x, context, ...) → x.

    Implementation: CrossAttention.
    """

    def __call__(
        self,
        x: Tensor,
        context: Tensor,
        x_seq_info: SequenceInfo | None = None,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Tensor: ...
