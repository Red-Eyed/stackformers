from __future__ import annotations

import pytest
import torch

from stackformers.presets.pyramid_vision import (
    PyramidVisionBackbone,
    pyramid_vision_config,
)
from stackformers.spatial.input import make_spatial_input

B = 2
# Small grid that still divides cleanly by window=4 and the sr_ratios — keeps the test fast.
DIMS = (16, 32, 64, 128)
HEADS = (1, 2, 4, 8)
GRID = (16, 16)


def _config(**kw: object):
    return pyramid_vision_config(
        dims=DIMS,
        depths=(1, 1, 1, 1),
        heads=HEADS,
        sr_ratios=(4, 2, 1, 1),
        window=4,
        window_stages=2,
        **kw,  # type: ignore[arg-type]
    )


def test_backbone_emits_one_feature_map_per_stage(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    backbone = PyramidVisionBackbone(_config()).to(device=device, dtype=dtype)
    x = torch.randn(B, GRID[0] * GRID[1], DIMS[0], device=device, dtype=dtype)
    feats = backbone(make_spatial_input(x, GRID))
    assert len(feats) == len(DIMS)


def test_backbone_feature_maps_halve_and_widen(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    backbone = PyramidVisionBackbone(_config()).to(device=device, dtype=dtype)
    x = torch.randn(B, GRID[0] * GRID[1], DIMS[0], device=device, dtype=dtype)
    feats = backbone(make_spatial_input(x, GRID))
    for i, feat in enumerate(feats):
        scale = 2**i
        assert feat.shape == (B, DIMS[i], GRID[0] // scale, GRID[1] // scale)


def test_backbone_gradients(device: torch.device) -> None:
    backbone = PyramidVisionBackbone(_config()).to(device=device)
    x = torch.randn(B, GRID[0] * GRID[1], DIMS[0], device=device, requires_grad=True)
    feats = backbone(make_spatial_input(x, GRID))
    torch.stack([f.sum() for f in feats]).sum().backward()
    assert x.grad is not None


def test_config_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="out_dim"):
        cfg = _config()
        cfg.inter_stage[0].out_dim = 999
        cfg.model_validate(cfg.model_dump())


def test_config_round_trips_through_json() -> None:
    cfg = _config()
    restored = type(cfg).model_validate_json(cfg.model_dump_json())
    assert restored == cfg
