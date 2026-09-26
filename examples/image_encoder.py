"""Classify small images with patch tokens and two-dimensional rotary positions."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from typing_extensions import override

from stackformers import (
    PaddedInput,
    RoPE2DConfig,
    TransformerEncoder,
    plain_encoder_config,
)

DIM = 64
IMAGE_SIZE = 16
PATCH_SIZE = 4
NUM_CLASSES = 10


class ImageEncoderResult(NamedTuple):
    """Image-token and classifier shapes produced by :func:`run_example`."""

    patch_tokens: int
    logits_shape: tuple[int, ...]


class ImagePatchEncoder(nn.Module):
    """Project image patches, encode their grid positions, and classify pooled features."""

    def __init__(self) -> None:
        """Build a patch projection, RoPE-2D encoder, and classification head."""
        super().__init__()
        self.patch_projection = nn.Conv2d(3, DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        config = plain_encoder_config(dim=DIM, heads=1, num_layers=2, ff_mult=3.0)
        config = config.model_copy(update={"pos_encoding": RoPE2DConfig(dim_head=DIM)})
        self.encoder = TransformerEncoder(config)
        self.classifier = nn.Linear(DIM, NUM_CLASSES)

    def _patch_input(self, images: Tensor) -> PaddedInput:
        """Convert a dense image batch into tokens with integer row/column coordinates."""
        patches = self.patch_projection(images)
        batch, _, rows, columns = patches.shape
        tokens = rearrange(patches, "b d r c -> b (r c) d")
        row_ids, column_ids = torch.meshgrid(
            torch.arange(rows, device=images.device),
            torch.arange(columns, device=images.device),
            indexing="ij",
        )
        positions = torch.stack((row_ids, column_ids), dim=-1).reshape(1, rows * columns, 2)
        positions = positions.to(dtype=tokens.dtype).expand(batch, -1, -1)
        mask = torch.ones(batch, rows * columns, dtype=torch.bool, device=images.device)
        return PaddedInput(x=tokens, mask=mask, abs_positions=positions)

    @override
    def forward(self, images: Tensor) -> Tensor:
        """Return one class-logit vector per image."""
        encoded = self.encoder(self._patch_input(images))
        output: Tensor = self.classifier(encoded.mean(dim=1))
        return output

    if TYPE_CHECKING:
        __call__ = forward


def run_example() -> ImageEncoderResult:
    """Encode a deterministic image batch and report its output geometry."""
    torch.manual_seed(0)
    model = ImagePatchEncoder().eval()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(images)
    patch_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    return ImageEncoderResult(patch_tokens=patch_tokens, logits_shape=tuple(logits.shape))


def main() -> None:
    """Print the image patch count and classifier output shape."""
    result = run_example()
    print(f"patch tokens per image: {result.patch_tokens}")
    print(f"logits: {result.logits_shape}")


if __name__ == "__main__":
    main()
