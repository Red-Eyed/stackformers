from __future__ import annotations

from einops import rearrange, repeat
from torch import Tensor

from stackformers.attention.config import AttentionConfig
from stackformers.attention.protocols import AttnKernel
from stackformers.attention.self_attn import BaseSelfAttention
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import PackedInput, PackedSequence


class PackedSelfAttention(BaseSelfAttention):
    """Packed multi-head self-attention — trains on variable-length sequences without padding.

    Shares all parameters with SelfAttention: state dicts are interchangeable
    for switching between training (packed) and inference/export (padded).
    """

    def __init__(
        self,
        config: AttentionConfig,
        pos_encoding: PosEncoding,
        kernel: AttnKernel,
    ) -> None:
        super().__init__(config)
        self.pos_encoding = pos_encoding
        self.kernel = kernel

    def forward(self, input: PackedInput) -> Tensor:
        h, kv_h, groups = self.config.heads, self.config.effective_kv_heads, self.config.groups

        x = input.x
        q = rearrange(self.to_q(x), "nt (h d) -> nt h d", h=h)
        k = rearrange(self.to_k(x), "nt (h d) -> nt h d", h=kv_h)
        v = rearrange(self.to_v(x), "nt (h d) -> nt h d", h=kv_h)

        if groups > 1:
            k = repeat(k, "nt h d -> nt (h g) d", g=groups)
            v = repeat(v, "nt h d -> nt (h g) d", g=groups)

        q, k = self.pos_encoding.forward(q, k, input, input)
        seq_info = PackedSequence(cu_seqlens=input.cu_seqlens, max_seqlen=input.max_seqlen)
        out = self.kernel.forward(q, k, v, seq_info, seq_info)

        return self.to_out(rearrange(out, "nt h d -> nt (h d)"))
