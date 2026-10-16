"""Self-attention transformer layers for each normalization topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn
from typing_extensions import override

from stackformers.attention.layout import call_self_attention
from stackformers.norm.config import NormPlacement as NormPlacement

if TYPE_CHECKING:
    from stackformers.attention.masking import SelfAttentionMask
    from stackformers.attention.protocols import SelfAttn
    from stackformers.feedforward.protocols import FeedForward
    from stackformers.norm.protocols import Norm
    from stackformers.sequence import SequenceInput


class TransformerLayerBase(nn.Module, ABC):
    """Common contract and branch modules for transformer-layer topologies."""

    def __init__(self, self_attn: SelfAttn, ff: FeedForward) -> None:
        """Store the attention and feed-forward branches shared by every topology."""
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff

    @abstractmethod
    @override
    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> SequenceInput:
        """Apply one transformer layer while preserving sequence metadata."""

    if TYPE_CHECKING:
        __call__ = forward


class TransformerLayer(TransformerLayerBase):
    """Pre-norm transformer layer preserving the original stackformers topology.

    Reference: Xiong et al., "On Layer Normalization in the Transformer Architecture"
    (ICML 2020), https://proceedings.mlr.press/v119/xiong20b.html.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        ff: FeedForward,
        norm_attn: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a layer that normalizes each branch input."""
        super().__init__(self_attn, ff)
        self.norm_attn = norm_attn
        self.norm_ff = norm_ff

    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> SequenceInput:
        """Apply pre-norm attention and feed-forward residual branches."""
        normalized_self = self.norm_attn(input.x)
        self_input = input._replace(x=normalized_self)
        self_out = call_self_attention(self.self_attn, self_input, self_attn_mask=self_attn_mask)
        x = input.x + self_out

        normalized_ff = self.norm_ff(x)
        ff_out = self.ff(normalized_ff)
        x = x + ff_out
        return input._replace(x=x)

    if TYPE_CHECKING:
        __call__ = forward


class PostNormTransformerLayer(TransformerLayerBase):
    """Post-norm layer that normalizes each completed residual sum.

    Reference: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017),
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        ff: FeedForward,
        norm_attn: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a layer with one residual-sum norm per branch."""
        super().__init__(self_attn, ff)
        self.norm_attn = norm_attn
        self.norm_ff = norm_ff

    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> SequenceInput:
        """Apply each norm after its branch has been added to the residual."""
        self_out = call_self_attention(self.self_attn, input, self_attn_mask=self_attn_mask)
        x = input.x + self_out
        x = self.norm_attn(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm_ff(x)
        return input._replace(x=x)

    if TYPE_CHECKING:
        __call__ = forward


class SandwichNormTransformerLayer(TransformerLayerBase):
    """Sandwich-norm layer with independent norms on both sides of each branch.

    Reference: Ding et al., "CogView: Mastering Text-to-Image Generation via Transformers"
    (NeurIPS 2021), https://arxiv.org/abs/2105.13290.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        ff: FeedForward,
        norm_attn_pre: Norm,
        norm_attn_post: Norm,
        norm_ff_pre: Norm,
        norm_ff_post: Norm,
    ) -> None:
        """Build a layer from four independently trainable branch norms."""
        super().__init__(self_attn, ff)
        self.norm_attn_pre = norm_attn_pre
        self.norm_attn_post = norm_attn_post
        self.norm_ff_pre = norm_ff_pre
        self.norm_ff_post = norm_ff_post

    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> SequenceInput:
        """Normalize the input and output of each residual branch."""
        normalized_self = self.norm_attn_pre(input.x)
        self_input = input._replace(x=normalized_self)
        self_out = call_self_attention(self.self_attn, self_input, self_attn_mask=self_attn_mask)
        self_branch = self.norm_attn_post(self_out)
        x = input.x + self_branch

        normalized_ff = self.norm_ff_pre(x)
        ff_out = self.ff(normalized_ff)
        ff_branch = self.norm_ff_post(ff_out)
        x = x + ff_branch
        return input._replace(x=x)

    if TYPE_CHECKING:
        __call__ = forward


class ReorderedNormTransformerLayer(TransformerLayerBase):
    """OLMo 2-style layer that normalizes branch outputs before residual addition.

    References: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution"
    (CVPR 2022), https://arxiv.org/abs/2111.09883; OLMo Team et al., "2 OLMo 2 Furious"
    (2025), https://arxiv.org/abs/2501.00656.
    """

    def __init__(
        self,
        self_attn: SelfAttn,
        ff: FeedForward,
        norm_attn: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a layer with one output norm per residual branch."""
        super().__init__(self_attn, ff)
        self.norm_attn = norm_attn
        self.norm_ff = norm_ff

    def forward(
        self, input: SequenceInput, *, self_attn_mask: SelfAttentionMask = "default"
    ) -> SequenceInput:
        """Normalize each branch result immediately before adding its residual."""
        self_out = call_self_attention(self.self_attn, input, self_attn_mask=self_attn_mask)
        self_branch = self.norm_attn(self_out)
        x = input.x + self_branch

        ff_out = self.ff(x)
        ff_branch = self.norm_ff(ff_out)
        x = x + ff_branch
        return input._replace(x=x)

    if TYPE_CHECKING:
        __call__ = forward
