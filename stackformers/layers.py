from __future__ import annotations

import torch.nn as nn

from stackformers.attention.protocols import SelfAttn
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.sequence import SequenceInput


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: norm → self-attn → residual, norm → ff → residual."""

    def __init__(
        self,
        self_attn: SelfAttn,
        ff: FeedForward,
        norm_attn: Norm,
        norm_ff: Norm,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.norm_attn = norm_attn
        self.norm_ff = norm_ff

    def forward(self, input: SequenceInput) -> SequenceInput:
        normed = input._replace(x=self.norm_attn(input.x))
        x = input.x + self.self_attn(normed)
        x = x + self.ff(self.norm_ff(x))
        return input._replace(x=x)
