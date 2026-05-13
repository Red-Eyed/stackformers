from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.sequence import SequenceInfo


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: norm → self-attn → residual, norm → ff → residual."""

    def __init__(
        self,
        self_attn: SelfAttention,
        ff: SwiGLU,
        norm_attn: RMSNorm,
        norm_ff: RMSNorm,
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
