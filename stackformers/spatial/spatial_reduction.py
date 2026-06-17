from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from stackformers.attention.ops import padded_sdpa
from stackformers.positional.protocols import PosEncoding
from stackformers.spatial.config import SpatialReductionAttentionConfig
from stackformers.spatial.input import SpatialInput
from stackformers.spatial.protocols import KVReduction


class SpatialReductionAttention(nn.Module):
    """Global self-attention with a spatially-reduced key/value context.

    Queries come from the full grid; keys/values from the injected KVReduction, which
    pools the grid to a coarse token set. Cost is O(n·s) with s ≪ n. Still a dot-product
    attention, so RoPE 2-D, GQA, and qk_norm all apply.

    Paper: "Pyramid Vision Transformer" (spatial-reduction attention), Wang et al., 2021 —
    https://arxiv.org/abs/2102.12122
    """

    def __init__(
        self,
        config: SpatialReductionAttentionConfig,
        pos_encoding: PosEncoding,
        kv_reduction: KVReduction,
    ) -> None:
        super().__init__()
        self.config = config
        h, kv_h, dh = config.heads, config.effective_kv_heads, config.dim_head
        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_encoding = pos_encoding
        self.kv_reduction = kv_reduction
        self.q_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        nn.init.normal_(self.to_out.weight, std=0.02)

    def forward(self, input: SpatialInput) -> Float[Tensor, "b n d"]:
        """Shape flow (n = H·W queries, s = reduced K/V tokens, s ≪ n):

        x        (b, n, d)         full grid
        reduce   ctx (b, s, d)     K/V context pooled to a coarse grid
        q        (b, h, n, dh)     queries from every grid cell
        k,v      (b, h, s, dh)     keys/values from the reduced context
        sdpa     (b, h, n, dh)     each query attends all s reduced tokens
        out      (b, n, d)         merge heads, output projection
        """
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        x = input.x
        ctx, kv_pos = self.kv_reduction(x, input.grid)
        q = self.q_norm(rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h))
        k = self.k_norm(rearrange(self.to_k(ctx), "b s (h d) -> b h s d", h=kv_h))
        v = rearrange(self.to_v(ctx), "b s (h d) -> b h s d", h=kv_h)
        if groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=groups)
        q, k = self.pos_encoding.forward_padded(q, k, input.abs_positions, kv_pos)
        kv_mask = torch.ones(ctx.shape[0], ctx.shape[1], dtype=torch.bool, device=ctx.device)
        out = padded_sdpa(q, k, v, kv_mask, causal=False, window_size=None, bias=None)
        return self.dropout(self.to_out(rearrange(out, "b h n d -> b n (h d)")))
