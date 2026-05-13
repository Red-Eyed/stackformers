from __future__ import annotations

import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.protocols import AttnBiasBuilder, AttnKernel
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import SequenceInfo


class BaseSelfAttention(nn.Module):
    """Owns all learnable parameters for self-attention.

    Subclass for padded (SelfAttention) or packed (PackedSelfAttention) forward paths.
    State dicts are compatible between subclasses — parameter names are identical.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        h, kv_h, dh = config.heads, config.effective_kv_heads, config.dim_head
        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        nn.init.normal_(self.to_out.weight, std=0.02)


class SelfAttention(BaseSelfAttention):
    """Padded multi-head self-attention with injected kernel, pos-encoding, and bias builder."""

    def __init__(
        self,
        config: AttentionConfig,
        pos_encoding: PosEncoding,
        bias_builder: AttnBiasBuilder,
        kernel: AttnKernel,
    ) -> None:
        super().__init__(config)
        self.pos_encoding = pos_encoding
        self.bias_builder = bias_builder
        self.kernel = kernel

    def forward(self, x: Tensor, seq_info: SequenceInfo) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups
        _, n, _ = x.shape

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=kv_h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=kv_h)

        if groups > 1:
            k = repeat(k, "b h n d -> b (h g) n d", g=groups)
            v = repeat(v, "b h n d -> b (h g) n d", g=groups)

        q, k = self.pos_encoding.forward(q, k, seq_info, seq_info)
        attn_bias = self.bias_builder.forward(n, n, x.device)
        out = self.kernel.forward(q, k, v, seq_info, seq_info, attn_bias)

        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))
