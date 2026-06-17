from __future__ import annotations

import pytest
import torch

from stackformers.spatial.input import SpatialInput, grid_positions, make_spatial_input

B, H, W, D = 2, 4, 6, 8
N = H * W


def test_make_spatial_input_shapes(device: torch.device) -> None:
    x = torch.randn(B, N, D, device=device)
    inp = make_spatial_input(x, grid=(H, W))
    assert isinstance(inp, SpatialInput)
    assert inp.x.shape == (B, N, D)
    assert inp.mask.shape == (B, N)
    assert inp.abs_positions.shape == (B, N, 2)
    assert inp.grid == (H, W)


def test_grid_positions_are_row_major(device: torch.device) -> None:
    pos = grid_positions(H, W, device)
    assert pos.shape == (N, 2)
    # first row spans cols 0..W-1 at row 0
    assert torch.equal(pos[:W, 0], torch.zeros(W, device=device))
    assert torch.equal(pos[:W, 1], torch.arange(W, device=device, dtype=torch.float32))
    # second token-row starts at grid row 1
    assert pos[W, 0].item() == 1.0


def test_make_spatial_input_rejects_mismatched_grid(device: torch.device) -> None:
    x = torch.randn(B, N + 1, D, device=device)
    with pytest.raises(ValueError, match="implies"):
        make_spatial_input(x, grid=(H, W))


def test_make_spatial_input_all_valid_mask(device: torch.device) -> None:
    x = torch.randn(B, N, D, device=device)
    inp = make_spatial_input(x, grid=(H, W))
    assert bool(inp.mask.all())
