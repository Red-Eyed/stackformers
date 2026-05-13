from __future__ import annotations

from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.cross_attn import BaseCrossAttention
from stackformers.attention.protocols import AttnKernel
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PackedSequence


class PackedCrossAttention(BaseCrossAttention):
    """Packed multi-head cross-attention: queries from x attend to packed context.

    Shares all parameters with CrossAttention: state dicts are interchangeable
    for switching between training (packed) and inference/export (padded).
    No bias builder — packed kernels do not support additive attention bias.
    """

    def __init__(
        self,
        config,
        pos_encoding: PosEncoding,
        kernel: AttnKernel,
    ) -> None:
        super().__init__(config)
        self.pos_encoding = pos_encoding
        self.kernel = kernel

    def forward(
        self,
        x_input: PackedInput,
        ctx_input: PackedInput,
    ) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups

        x, context = x_input.x, ctx_input.x
        q = rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h)
        k = rearrange(self.to_k(context), "nt (h d) -> nt h d", h=kv_h)
        v = rearrange(self.to_v(context), "nt (h d) -> nt h d", h=kv_h)

        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)

        q, k = self.pos_encoding.forward(q, k, x_input, ctx_input)
        x_seq = PackedSequence(cu_seqlens=x_input.cu_seqlens, max_seqlen=x_input.max_seqlen)
        ctx_seq = PackedSequence(cu_seqlens=ctx_input.cu_seqlens, max_seqlen=ctx_input.max_seqlen)
        out = self.kernel.forward(q, k, v, x_seq, ctx_seq, None)

        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))
