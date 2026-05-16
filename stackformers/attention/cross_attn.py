from __future__ import annotations

import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.protocols import AttnKernel
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput, to_seq_info


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from x, keys/values from context.

    Accepts PaddedInput (inference) or PackedInput (training) for both x and context.
    """

    def __init__(
        self,
        config: AttentionConfig,
        pos_encoding: PosEncoding,
        kernel: AttnKernel,
    ) -> None:
        super().__init__()
        self.config = config
        h, kv_h, dh = config.heads, config.effective_kv_heads, config.dim_head
        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.pos_encoding = pos_encoding
        self.kernel = kernel
        nn.init.normal_(self.to_out.weight, std=0.02)

    def _forward_padded(self, x_input: PaddedInput, ctx_input: PaddedInput) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups
        x, context = x_input.x, ctx_input.x
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(context), "b s (h d) -> b h s d", h=kv_h)
        v = rearrange(self.to_v(context), "b s (h d) -> b h s d", h=kv_h)
        if groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=groups)
        q, k = self.pos_encoding.forward(q, k, x_input, ctx_input)
        out = self.kernel.forward(q, k, v, to_seq_info(x_input), to_seq_info(ctx_input))
        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return out * x_input.mask.unsqueeze(-1)

    def _forward_packed(self, x_input: PackedInput, ctx_input: PackedInput) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups
        x, context = x_input.x, ctx_input.x
        q = rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h)
        k = rearrange(self.to_k(context), "nt (h d) -> nt h d", h=kv_h)
        v = rearrange(self.to_v(context), "nt (h d) -> nt h d", h=kv_h)
        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)
        q, k = self.pos_encoding.forward(q, k, x_input, ctx_input)
        out = self.kernel.forward(q, k, v, to_seq_info(x_input), to_seq_info(ctx_input))
        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        match x_input:
            case PaddedInput():
                return self._forward_padded(x_input, ctx_input)  # type: ignore[arg-type]
            case PackedInput():
                return self._forward_packed(x_input, ctx_input)  # type: ignore[arg-type]
