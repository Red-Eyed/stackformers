from __future__ import annotations

import torch.nn as nn
from einops import rearrange

from stackformers.norm.protocols import Norm
from stackformers.spatial.config import PatchMergingConfig
from stackformers.spatial.input import SpatialInput, make_spatial_input


def _out_len(size: int, kernel: int, stride: int, padding: int) -> int:
    """Conv output length along one spatial axis (static, from Python ints)."""
    return (size + 2 * padding - kernel) // stride + 1


class PatchMerging(nn.Module):
    """Downsample a grid between pyramid stages with an overlapping strided conv (PVTv2).

    Maps SpatialInput → SpatialInput: a stride-s conv shrinks the grid and widens the
    channels, then the injected Norm normalises. The new grid is derived from Python ints
    so the output shape stays static for torch.export.
    """

    def __init__(self, config: PatchMergingConfig, norm: Norm) -> None:
        super().__init__()
        self.kernel = config.kernel
        self.stride = config.stride
        self.padding = config.padding
        self.conv = nn.Conv2d(
            config.in_dim,
            config.out_dim,
            kernel_size=config.kernel,
            stride=config.stride,
            padding=config.padding,
        )
        self.norm = norm

    def forward(self, input: SpatialInput) -> SpatialInput:
        """Shape flow (stride s, kernel k, padding p):

        x        (b, n, in_dim)        n = H·W
        to grid  (b, in_dim, H, W)
        conv     (b, out_dim, H', W')  H' = (H+2p-k)//s + 1
        out      (b, n', out_dim)      n' = H'·W', new grid (H', W')
        """
        h, w = input.grid
        feat = rearrange(input.x, "b (h w) d -> b d h w", h=h, w=w)
        feat = self.conv(feat)
        new_grid = (
            _out_len(h, self.kernel, self.stride, self.padding),
            _out_len(w, self.kernel, self.stride, self.padding),
        )
        x = self.norm(rearrange(feat, "b d h w -> b (h w) d"))
        return make_spatial_input(x, new_grid)
