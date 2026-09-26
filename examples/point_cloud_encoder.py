"""Encode continuous 3-D point coordinates with N-dimensional rotary positions."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers import PaddedInput, RoPENDConfig, TransformerEncoder, plain_encoder_config

DIM = 192
INPUT_FEATURES = 6


class PointCloudResult(NamedTuple):
    """Output geometry and translation agreement produced by :func:`run_example`."""

    output_shape: tuple[int, ...]
    translation_max_abs_error: float


def _center_coordinates(coordinates: Tensor, mask: Tensor) -> Tensor:
    """Center each point set using only valid points to preserve numerical precision."""
    valid = mask.unsqueeze(-1)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    centroid = (coordinates * valid).sum(dim=1, keepdim=True) / counts
    return coordinates - centroid


class PointCloudEncoder(nn.Module):
    """Project point features and apply direction-aware attention over 3-D coordinates."""

    def __init__(self) -> None:
        """Build a feature projection and a continuous-coordinate RoPE encoder."""
        super().__init__()
        self.feature_projection = nn.Linear(INPUT_FEATURES, DIM)
        config = plain_encoder_config(dim=DIM, heads=1, num_layers=2, ff_mult=1.0)
        config = config.model_copy(
            update={
                "pos_encoding": RoPENDConfig(
                    dim_head=DIM,
                    coords=3,
                    r_min=0.25,
                    r_max=8.0,
                )
            }
        )
        self.encoder = TransformerEncoder(config)

    @override
    def forward(self, features: Tensor, coordinates: Tensor, mask: Tensor) -> Tensor:
        """Return one encoded vector per point while retaining the input layout."""
        x = self.feature_projection(features)
        positions = _center_coordinates(coordinates, mask)
        return self.encoder(PaddedInput(x=x, mask=mask, abs_positions=positions))

    if TYPE_CHECKING:
        __call__ = forward


def run_example() -> PointCloudResult:
    """Verify that a global coordinate translation leaves encoded points unchanged."""
    torch.manual_seed(0)
    model = PointCloudEncoder().eval()
    features = torch.randn(2, 5, INPUT_FEATURES)
    coordinates = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    translation = torch.tensor([100.0, -50.0, 25.0])
    with torch.no_grad():
        original = model(features, coordinates, mask)
        translated = model(features, coordinates + translation, mask)
    error = float((original - translated).abs().max())
    return PointCloudResult(output_shape=tuple(original.shape), translation_max_abs_error=error)


def main() -> None:
    """Print the point output shape and translation agreement."""
    result = run_example()
    print(f"encoded points: {result.output_shape}")
    print(f"translation maximum absolute error: {result.translation_max_abs_error:.3e}")


if __name__ == "__main__":
    main()
