from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.v1.attention.protocols import SelfAttn
from stackformers.v1.feedforward.protocols import FeedForward
from stackformers.v1.norm.protocols import Norm
from stackformers.v1.sequence import SequenceInfo


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

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        seq_info: SequenceInfo,
    ) -> Float[Tensor, "b n d"]:
        x = x + self.self_attn(self.norm_attn(x), seq_info)
        x = x + self.ff(self.norm_ff(x))
        return x
