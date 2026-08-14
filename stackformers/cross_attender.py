"""Cross-attender layers and stacks for each normalization topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor

from stackformers.attention.protocols import CrossAttn
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.sequence import SequenceInput


class CrossAttenderLayerBase(nn.Module, ABC):
    """Common contract and branches for cross-attender normalization topologies."""

    def __init__(self, cross_attn: CrossAttn, ff: FeedForward) -> None:
        """Store the cross-attention and feed-forward residual branches."""
        super().__init__()
        self.cross_attn = cross_attn
        self.ff = ff

    @abstractmethod
    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        """Apply one cross-attender layer while preserving query metadata."""


class CrossAttenderLayer(CrossAttenderLayerBase):
    """Pre-norm cross-attender layer.

    Reference: Xiong et al., "On Layer Normalization in the Transformer Architecture"
    (ICML 2020), https://proceedings.mlr.press/v119/xiong20b.html.
    """

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a cross-attender that normalizes each branch input."""
        super().__init__(cross_attn, ff)
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        """Apply pre-norm cross-attention and feed-forward residual branches."""
        normed = x_input._replace(x=self.norm_cross(x_input.x))
        x = x_input.x + self.cross_attn(normed, ctx_input)
        x = x + self.ff(self.norm_ff(x))
        return x_input._replace(x=x)


class PostNormCrossAttenderLayer(CrossAttenderLayerBase):
    """Post-norm cross-attender that normalizes each completed residual sum.

    Reference: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017),
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a cross-attender with one residual-sum norm per branch."""
        super().__init__(cross_attn, ff)
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        """Normalize each residual sum before evaluating the next branch."""
        x = self.norm_cross(x_input.x + self.cross_attn(x_input, ctx_input))
        x = self.norm_ff(x + self.ff(x))
        return x_input._replace(x=x)


class SandwichNormCrossAttenderLayer(CrossAttenderLayerBase):
    """Sandwich-norm cross-attender with independent norms around both branches.

    Reference: Ding et al., "CogView: Mastering Text-to-Image Generation via Transformers"
    (NeurIPS 2021), https://arxiv.org/abs/2105.13290.
    """

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross_pre: Norm,
        norm_cross_post: Norm,
        norm_ff_pre: Norm,
        norm_ff_post: Norm,
    ) -> None:
        """Build a cross-attender from four independently trainable norms."""
        super().__init__(cross_attn, ff)
        self.norm_cross_pre = norm_cross_pre
        self.norm_cross_post = norm_cross_post
        self.norm_ff_pre = norm_ff_pre
        self.norm_ff_post = norm_ff_post

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        """Normalize the input and output of both residual branches."""
        normed = x_input._replace(x=self.norm_cross_pre(x_input.x))
        x = x_input.x + self.norm_cross_post(self.cross_attn(normed, ctx_input))
        x = x + self.norm_ff_post(self.ff(self.norm_ff_pre(x)))
        return x_input._replace(x=x)


class ReorderedNormCrossAttenderLayer(CrossAttenderLayerBase):
    """Residual-post-norm cross-attender that normalizes branch outputs.

    References: Liu et al., "Swin Transformer V2" (CVPR 2022),
    https://arxiv.org/abs/2111.09883; OLMo Team et al., "2 OLMo 2 Furious" (2025),
    https://arxiv.org/abs/2501.00656.
    """

    def __init__(
        self,
        cross_attn: CrossAttn,
        ff: FeedForward,
        norm_cross: Norm,
        norm_ff: Norm,
    ) -> None:
        """Build a cross-attender with one output norm per residual branch."""
        super().__init__(cross_attn, ff)
        self.norm_cross = norm_cross
        self.norm_ff = norm_ff

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> SequenceInput:
        """Normalize each branch result immediately before residual addition."""
        x = x_input.x + self.norm_cross(self.cross_attn(x_input, ctx_input))
        x = x + self.norm_ff(self.ff(x))
        return x_input._replace(x=x)


class CrossAttenderStack(nn.Module):
    """Stack cross-attender layers and apply a final normalization."""

    def __init__(self, layers: Sequence[CrossAttenderLayerBase], final_norm: Norm) -> None:
        """Register the ordered cross-attender layers and final norm."""
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        """Transform a query sequence against a fixed context sequence."""
        for layer in self.layers:
            x_input = layer(x_input, ctx_input)
        return self.final_norm(x_input.x)
