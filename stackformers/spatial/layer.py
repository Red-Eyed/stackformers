from __future__ import annotations

import torch.nn as nn

from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.protocols import Norm
from stackformers.spatial.input import SpatialInput
from stackformers.spatial.protocols import SpatialAttn


class SpatialTransformerLayer(nn.Module):
    """Pre-norm transformer layer over a 2-D grid: norm → spatial-attn → residual,
    norm → ff → residual.

    Mirrors TransformerLayer but is typed to SpatialAttn / SpatialInput so the grid shape
    rides through unchanged via NamedTuple._replace. The feed-forward and norms are the same
    injected protocols used everywhere else.
    """

    def __init__(
        self,
        attn: SpatialAttn,
        ff: FeedForward,
        norm_attn: Norm,
        norm_ff: Norm,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.ff = ff
        self.norm_attn = norm_attn
        self.norm_ff = norm_ff

    def forward(self, input: SpatialInput) -> SpatialInput:
        """Shape-invariant: x stays (b, n, d) throughout; grid rides along via _replace.

        x = x + attn(norm(x))   # (b, n, d) → (b, n, d)
        x = x + ff(norm(x))     # (b, n, d) → (b, n, d)
        """
        normed = input._replace(x=self.norm_attn(input.x))
        x = input.x + self.attn(normed)
        x = x + self.ff(self.norm_ff(x))
        return input._replace(x=x)
