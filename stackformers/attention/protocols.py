from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from stackformers.sequence import SequenceInfo


@runtime_checkable
class AttnKernel(Protocol):
    """Compute scaled dot-product attention given projected heads.

    Implementations: SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel,
    VarlenWindowedSDPAKernel.
    """

    def forward(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        v: Float[Tensor, "b h s dh"],
        attn_mask: Float[Tensor, "b 1 n s"] | None,
        attn_bias: Float[Tensor, "h n s"] | None,
        is_causal: bool,
    ) -> Float[Tensor, "b h n dh"]: ...


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
        device: torch.device,
    ) -> Float[Tensor, "h n s"] | None: ...


@runtime_checkable
class SelfAttn(Protocol):
    """Self-attention over a sequence: maps (x, seq_info) → x.

    Implementation: SelfAttention.
    """

    def __call__(
        self,
        x: Float[Tensor, "b n d"],
        seq_info: SequenceInfo,
    ) -> Float[Tensor, "b n d"]: ...


@runtime_checkable
class CrossAttn(Protocol):
    """Cross-attention from x to context: maps (x, context, ctx_seq_info) → x.

    Implementation: CrossAttention.
    """

    def __call__(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b s d"],
        x_seq_info: SequenceInfo | None = None,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]: ...
