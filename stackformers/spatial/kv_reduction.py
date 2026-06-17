from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from stackformers.norm.protocols import Norm
from stackformers.spatial.config import ConvKVReductionConfig
from stackformers.spatial.input import grid_positions


class NoKVReduction(nn.Module):
    """Null object for KVReduction — returns the full grid unchanged.

    Keeps the attention forward() free of `if reduction is not None` branches.
    """

    def __call__(
        self,
        x: Float[Tensor, "b n d"],
        grid: tuple[int, int],
    ) -> tuple[Float[Tensor, "b s d"], Float[Tensor, "b s c"]]:
        h, w = grid
        positions = grid_positions(h, w, x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        return x, positions


class ConvKVReduction(nn.Module):
    """Downsample K/V context with a strided conv, then norm.

    Reshapes the flat token grid to (b, d, H, W), applies a stride-r conv that pools each
    r×r block to one token, flattens back, and normalises with the injected Norm. Emits the
    coarse grid's (row, col) positions so the caller can position-encode the reduced keys.

    Paper: "Pyramid Vision Transformer" (spatial-reduction attention), Wang et al., 2021 —
    https://arxiv.org/abs/2102.12122
    """

    def __init__(self, config: ConvKVReductionConfig, norm: Norm) -> None:
        super().__init__()
        self.reduction = config.reduction
        self.conv = nn.Conv2d(
            config.dim, config.dim, kernel_size=config.reduction, stride=config.reduction
        )
        self.norm = norm

    def __call__(
        self,
        x: Float[Tensor, "b n d"],
        grid: tuple[int, int],
    ) -> tuple[Float[Tensor, "b s d"], Float[Tensor, "b s c"]]:
        """Shape flow (r = reduction, s = (H/r)·(W/r)):

        x        (b, n, d)      n = H·W
        to grid  (b, d, H, W)
        conv     (b, d, H/r, W/r)
        ctx      (b, s, d)      flattened + normed reduced tokens
        pos      (b, s, 2)      coarse-grid (row, col)
        """
        h, w = grid
        r = self.reduction
        if h % r != 0 or w % r != 0:
            raise ValueError(f"grid {grid} not divisible by reduction {r}")
        feat = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
        feat = self.conv(feat)
        ctx = self.norm(rearrange(feat, "b d h w -> b (h w) d"))
        positions = grid_positions(h // r, w // r, x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        return ctx, positions
