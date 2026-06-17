from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def grid_positions(h: int, w: int, device: torch.device) -> Float[Tensor, "n c"]:
    """Row-major (row, col) coordinates for an h×w grid, shape (h*w, 2).

    Built with static h, w so it stays torch.export / torch.compile friendly.
    """
    rows = torch.arange(h, device=device).repeat_interleave(w)
    cols = torch.arange(w, device=device).repeat(h)
    return torch.stack([rows, cols], dim=-1).to(torch.float32)


class SpatialInput(NamedTuple):
    """Padded batch of tokens laid out on a 2-D grid.

    Distinct from SequenceInput: it carries the grid shape so spatial attention can
    reshape the flat token sequence back to (b, d, H, W). Padded-only — there is no
    packed spatial variant (fixed-resolution images do not vary in length).

    NamedTuple (not dataclass) so torch.export / torch.compile see the tensor fields
    natively, matching PaddedInput / PackedInput.
    """

    x: Float[Tensor, "b n d"]
    mask: Bool[Tensor, "b n"]  # True = valid token
    abs_positions: Float[Tensor, "b n c"]  # c=2, (row, col) per token
    grid: tuple[int, int]  # (H, W); H * W == n


def make_spatial_input(
    x: Float[Tensor, "b n d"],
    grid: tuple[int, int],
    mask: Bool[Tensor, "b n"] | None = None,
) -> SpatialInput:
    """Build a SpatialInput with row-major (row, col) positions from the grid shape.

    mask defaults to all-valid; pass an explicit mask only when the batch is padded.
    """
    h, w = grid
    n = h * w
    if x.shape[1] != n:
        raise ValueError(f"x has {x.shape[1]} tokens but grid {grid} implies {n}")
    positions = grid_positions(h, w, x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
    if mask is None:
        mask = torch.ones(x.shape[0], n, dtype=torch.bool, device=x.device)
    return SpatialInput(x=x, mask=mask, abs_positions=positions, grid=grid)
