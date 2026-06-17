from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from stackformers.attention.ops import padded_sdpa
from stackformers.positional.protocols import PosEncoding
from stackformers.spatial.config import WindowAttention2DConfig
from stackformers.spatial.input import SpatialInput


def partition_windows(
    t: Tensor, h: int, w: int, window: int
) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    """Partition a row-major grid (b, n, c) into non-overlapping window×window tiles,
    shape ((b·nwin), window², c), and return an inverse that stitches tiles back to (b, n, c).

    The inverse is a closure bound to the exact (b, h, w, window) used here, so the caller
    cannot mismatch the merge parameters — the only way to un-partition is with this object.
    """
    b = t.shape[0]
    grid = rearrange(t, "b (h w) c -> b h w c", h=h, w=w)
    windows = rearrange(grid, "b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c", wh=window, ww=window)

    def merge(windowed: Tensor) -> Tensor:
        nh, nw = h // window, w // window
        regridded = rearrange(
            windowed,
            "(b nh nw) (wh ww) c -> b (nh wh) (nw ww) c",
            b=b,
            nh=nh,
            nw=nw,
            wh=window,
            ww=window,
        )
        return rearrange(regridded, "b h w c -> b (h w) c")

    return windows, merge


class WindowAttention2D(nn.Module):
    """Non-overlapping 2-D window self-attention — O(n·w²), local.

    Partitions the grid into window×window tiles, runs full attention inside each tile, and
    stitches the result back. RoPE 2-D uses each token's global (row, col) so relative
    position is correct across the window. Requires grid H, W divisible by the window.

    Window partitioning follows Swin (Liu et al., 2021, https://arxiv.org/abs/2103.14030);
    shifted windows are not implemented — that would be a separate variant.
    """

    def __init__(self, config: WindowAttention2DConfig, pos_encoding: PosEncoding) -> None:
        super().__init__()
        self.config = config
        self.window = config.window
        h, kv_h, dh = config.heads, config.effective_kv_heads, config.dim_head
        self.to_q = nn.Linear(config.dim, h * dh, bias=False)
        self.to_k = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_v = nn.Linear(config.dim, kv_h * dh, bias=False)
        self.to_out = nn.Linear(h * dh, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_encoding = pos_encoding
        self.q_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dh) if config.qk_norm else nn.Identity()
        nn.init.normal_(self.to_out.weight, std=0.02)

    def forward(self, input: SpatialInput) -> Float[Tensor, "b n d"]:
        """Shape flow (grid H×W, window w, nwin = (H/w)·(W/w), bw = b·nwin, wn = w²):

        x         (b, n, d)        n = H·W
        partition (bw, wn, d)      grid split into nwin tiles
        q,k,v     (bw, h, wn, dh)  per-tile heads
        sdpa      (bw, h, wn, dh)  full attention within each tile
        merge     (b, n, d)        tiles stitched back, then output projection
        """
        cfg = self.config
        h, kv_h, groups = cfg.heads, cfg.effective_kv_heads, cfg.groups
        gh, gw = input.grid
        win = self.window
        if gh % win != 0 or gw % win != 0:
            raise ValueError(f"grid {input.grid} not divisible by window {win}")

        xw, merge = partition_windows(input.x, gh, gw, win)  # (bw, wn, d)
        posw, _ = partition_windows(input.abs_positions, gh, gw, win)  # (bw, wn, 2)
        maskw, _ = partition_windows(input.mask[..., None], gh, gw, win)  # (bw, wn, 1)

        q = self.q_norm(rearrange(self.to_q(xw), "bw n (h d) -> bw h n d", h=h))
        k = self.k_norm(rearrange(self.to_k(xw), "bw n (h d) -> bw h n d", h=kv_h))
        v = rearrange(self.to_v(xw), "bw n (h d) -> bw h n d", h=kv_h)
        if groups > 1:
            k = repeat(k, "bw h n d -> bw (h g) n d", g=groups)
            v = repeat(v, "bw h n d -> bw (h g) n d", g=groups)
        q, k = self.pos_encoding.forward_padded(q, k, posw, posw)
        out = padded_sdpa(q, k, v, maskw[..., 0], causal=False, window_size=None, bias=None)

        out = merge(rearrange(out, "bw h n d -> bw n (h d)"))
        return self.dropout(self.to_out(out))
