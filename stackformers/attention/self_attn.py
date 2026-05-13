from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.protocols import AttnBiasBuilder, AttnKernel
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PaddedSequence, SequenceInfo


def _mask_to_float_bias(mask: Bool[Tensor, "b n"], dtype: torch.dtype) -> Float[Tensor, "b 1 1 n"]:
    """Convert bool mask (True=valid) to additive float bias for SDPA."""
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias = bias.masked_fill(~mask, torch.finfo(dtype).min)
    return rearrange(bias, "b s -> b 1 1 s")


class SelfAttention(nn.Module):
    """Multi-head self-attention with injected kernel, pos-encoding, and bias.

    All behavioral choices (causal/non-causal, RoPE, ALiBi, windowed) come
    from the injected collaborators — not from flags.
    """

    def __init__(
        self,
        config: AttentionConfig,
        pos_encoding: PosEncoding,
        bias_builder: AttnBiasBuilder,
        kernel: AttnKernel,
    ) -> None:
        super().__init__()
        self.config = config
        self.pos_encoding = pos_encoding
        self.bias_builder = bias_builder
        self.kernel = kernel

        h = config.heads
        kv_h = config.effective_kv_heads
        dh = config.dim_head

        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)

        nn.init.normal_(self.to_out.weight, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        seq_info: SequenceInfo,
    ) -> Float[Tensor, "b n d"]:
        b, n, _ = x.shape
        h = self.config.heads
        kv_h = self.config.effective_kv_heads
        groups = self.config.groups

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=kv_h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=kv_h)

        if groups > 1:
            k = repeat(k, "b h n d -> b (h g) n d", g=groups)
            v = repeat(v, "b h n d -> b (h g) n d", g=groups)

        q, k = self.pos_encoding.forward(q, k)

        attn_mask: Float[Tensor, "b 1 n n"] | None = None
        if isinstance(seq_info, PaddedSequence):
            attn_mask = _mask_to_float_bias(seq_info.mask, q.dtype)

        attn_bias = self.bias_builder.forward(n, n, x.device)

        out = self.kernel.forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=self.config.causal,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
