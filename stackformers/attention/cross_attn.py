from __future__ import annotations

import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.protocols import AttnBiasBuilder, AttnKernel
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PaddedSequence, SequenceInfo


class BaseCrossAttention(nn.Module):
    """Owns all learnable parameters for cross-attention.

    Subclass for padded (CrossAttention) or packed (PackedCrossAttention) forward paths.
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


class CrossAttention(BaseCrossAttention):
    """Padded multi-head cross-attention: queries from x, keys/values from context."""

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

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        x_seq_info: SequenceInfo | None = None,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups
        n, s = x.shape[1], context.shape[1]

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(context), "b s (h d) -> b h s d", h=kv_h)
        v = rearrange(self.to_v(context), "b s (h d) -> b h s d", h=kv_h)

        if groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=groups)

        q, k = self.pos_encoding.forward(q, k, x_seq_info, ctx_seq_info)
        attn_bias = self.bias_builder.forward(n, s, x.device)
        out = self.kernel.forward(q, k, v, x_seq_info or PaddedSequence(mask=x.new_ones(x.shape[0], n, dtype=bool)), ctx_seq_info, attn_bias)

        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))

        if isinstance(x_seq_info, PaddedSequence):
            out = out * x_seq_info.mask.unsqueeze(-1)

        return out
