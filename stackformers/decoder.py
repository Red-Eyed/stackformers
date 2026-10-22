"""Decoder layers and stacks for each supported normalization topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers._decoder_equations import (
    NoContext,
    WithContext,
    post_norm_decoder,
    pre_norm_decoder,
    reordered_norm_decoder,
    sandwich_norm_decoder,
)
from stackformers.attention.layout import call_cross_attention, call_self_attention
from stackformers.attention.masking import CrossAttentionMask, SelfAttentionMask, require_context

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stackformers.attention.protocols import CrossAttn, SelfAttn
    from stackformers.feedforward.protocols import FeedForward
    from stackformers.norm.protocols import Norm
    from stackformers.sequence import SequenceInput


def _ordinary_self_attention(
    attention: SelfAttn,
    input: SequenceInput,
    *,
    self_attn_mask: SelfAttentionMask,
) -> tuple[Tensor, tuple[()]]:
    """Adapt ordinary attention to the shared equation's explicit state-return contract."""
    output = call_self_attention(attention, input, self_attn_mask=self_attn_mask)
    return output, ()


class DecoderLayerBase(nn.Module, ABC):
    """Common contract and branches for decoder normalization topologies."""

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
    ) -> None:
        """Store the three residual branches shared by every decoder topology."""
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff

    @abstractmethod
    @override
    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> SequenceInput:
        """Decode one layer while preserving target-sequence metadata."""

    if TYPE_CHECKING:
        __call__ = forward


class DecoderLayer(DecoderLayerBase):
    """Pre-norm decoder layer.

    Reference: Xiong et al., "On Layer Normalization in the Transformer Architecture"
    (ICML 2020), https://proceedings.mlr.press/v119/xiong20b.html.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_self: Norm,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a decoder that normalizes each branch input."""
        super().__init__(self_attn, cross_attn, ff)
        self.norm_self = norm_self
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> SequenceInput:
        """Apply the shared pre-norm equation with ordinary attention."""
        require_context(ctx_input is not None, cross_attn_mask)
        context = NoContext() if ctx_input is None else WithContext(ctx_input)
        output, _ = pre_norm_decoder(
            x_input,
            partial(_ordinary_self_attention, self.self_attn, self_attn_mask=self_attn_mask),
            partial(call_cross_attention, self.cross_attn, cross_attn_mask=cross_attn_mask),
            context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )
        return output

    if TYPE_CHECKING:
        __call__ = forward


class PostNormDecoderLayer(DecoderLayerBase):
    """Post-norm decoder layer that normalizes each completed residual sum.

    Reference: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017),
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_self: Norm,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a decoder with one residual-sum norm per branch."""
        super().__init__(self_attn, cross_attn, ff)
        self.norm_self = norm_self
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> SequenceInput:
        """Apply the shared post-norm equation with ordinary attention."""
        require_context(ctx_input is not None, cross_attn_mask)
        context = NoContext() if ctx_input is None else WithContext(ctx_input)
        output, _ = post_norm_decoder(
            x_input,
            partial(_ordinary_self_attention, self.self_attn, self_attn_mask=self_attn_mask),
            partial(call_cross_attention, self.cross_attn, cross_attn_mask=cross_attn_mask),
            context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )
        return output

    if TYPE_CHECKING:
        __call__ = forward


class SandwichNormDecoderLayer(DecoderLayerBase):
    """Sandwich-norm decoder with independent norms around all three branches.

    Reference: Ding et al., "CogView: Mastering Text-to-Image Generation via Transformers"
    (NeurIPS 2021), https://arxiv.org/abs/2105.13290.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_self_pre: Norm,
        norm_self_post: Norm,
        norm_cross_pre: Norm,
        norm_cross_post: Norm,
        norm_ff_pre: Norm,
        norm_ff_post: Norm,
    ) -> None:
        """Build a decoder from six independently trainable branch norms."""
        super().__init__(self_attn, cross_attn, ff)
        self.norm_self_pre = norm_self_pre
        self.norm_self_post = norm_self_post
        self.norm_cross_pre = norm_cross_pre
        self.norm_cross_post = norm_cross_post
        self.norm_ff_pre = norm_ff_pre
        self.norm_ff_post = norm_ff_post

    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> SequenceInput:
        """Apply the shared sandwich-norm equation with ordinary attention."""
        require_context(ctx_input is not None, cross_attn_mask)
        context = NoContext() if ctx_input is None else WithContext(ctx_input)
        output, _ = sandwich_norm_decoder(
            x_input,
            partial(_ordinary_self_attention, self.self_attn, self_attn_mask=self_attn_mask),
            partial(call_cross_attention, self.cross_attn, cross_attn_mask=cross_attn_mask),
            context,
            self.ff,
            self.norm_self_pre,
            self.norm_self_post,
            self.norm_cross_pre,
            self.norm_cross_post,
            self.norm_ff_pre,
            self.norm_ff_post,
        )
        return output

    if TYPE_CHECKING:
        __call__ = forward


class ReorderedNormDecoderLayer(DecoderLayerBase):
    """Residual-post-norm decoder that normalizes branch outputs before addition.

    References: Liu et al., "Swin Transformer V2" (CVPR 2022),
    https://arxiv.org/abs/2111.09883; OLMo Team et al., "2 OLMo 2 Furious" (2025),
    https://arxiv.org/abs/2501.00656.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_self: Norm,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a decoder with one output norm per residual branch."""
        super().__init__(self_attn, cross_attn, ff)
        self.norm_self = norm_self
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> SequenceInput:
        """Apply the shared reordered-norm equation with ordinary attention."""
        require_context(ctx_input is not None, cross_attn_mask)
        context = NoContext() if ctx_input is None else WithContext(ctx_input)
        output, _ = reordered_norm_decoder(
            x_input,
            partial(_ordinary_self_attention, self.self_attn, self_attn_mask=self_attn_mask),
            partial(call_cross_attention, self.cross_attn, cross_attn_mask=cross_attn_mask),
            context,
            self.ff,
            self.norm_self,
            self.norm_cross,
            self.norm_ff,
        )
        return output

    if TYPE_CHECKING:
        __call__ = forward


class Decoder(nn.Module):
    """Stack decoder layers and apply a final normalization."""

    def __init__(self, layers: Sequence[DecoderLayerBase], final_norm: Norm) -> None:
        """Register the ordered decoder layers and final norm."""
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    @override
    def forward(
        self,
        x_input: SequenceInput,
        ctx_input: SequenceInput | None = None,
        *,
        self_attn_mask: SelfAttentionMask = "default",
        cross_attn_mask: CrossAttentionMask = None,
    ) -> Tensor:
        """Decode with optional context; omission skips the entire cross-attention branch."""
        require_context(ctx_input is not None, cross_attn_mask)
        for layer in self.layers:
            x_input = layer(
                x_input, ctx_input, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask
            )
        return self.final_norm(x_input.x)

    if TYPE_CHECKING:
        __call__ = forward
