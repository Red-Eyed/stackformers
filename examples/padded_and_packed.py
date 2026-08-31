"""Compare equivalent variable-length batches in padded and packed layouts."""

from __future__ import annotations

from typing import NamedTuple

import torch

from stackformers import (
    PackedInput,
    PaddedInput,
    TransformerEncoder,
    make_packed_input,
    make_padded_input,
    plain_encoder_config,
)

DIM = 64
SEQUENCE_LENGTHS = (4, 2)


class LayoutParityResult(NamedTuple):
    """Shapes and numerical agreement produced by :func:`run_example`."""

    padded_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    max_abs_error: float


def _equivalent_inputs() -> tuple[PaddedInput, PackedInput]:
    """Build padded and packed representations containing identical valid tokens."""
    generator = torch.Generator().manual_seed(1)
    valid_tokens = [torch.randn(length, DIM, generator=generator) for length in SEQUENCE_LENGTHS]
    padded_x = torch.zeros(len(SEQUENCE_LENGTHS), max(SEQUENCE_LENGTHS), DIM)
    mask = torch.zeros(len(SEQUENCE_LENGTHS), max(SEQUENCE_LENGTHS), dtype=torch.bool)
    for batch_index, tokens in enumerate(valid_tokens):
        padded_x[batch_index, : tokens.shape[0]] = tokens
        mask[batch_index, : tokens.shape[0]] = True

    packed_x = torch.cat(valid_tokens)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    return (
        make_padded_input(padded_x, mask),
        make_packed_input(packed_x, cu_seqlens, max_seqlen=max(SEQUENCE_LENGTHS)),
    )


def run_example() -> LayoutParityResult:
    """Run one encoder over both layouts and compare outputs at valid positions."""
    torch.manual_seed(0)
    model = TransformerEncoder(
        plain_encoder_config(dim=DIM, heads=1, num_layers=2, ff_mult=3.0)
    ).eval()
    padded, packed = _equivalent_inputs()
    with torch.no_grad():
        padded_output = model(padded)
        packed_output = model(packed)

    valid_padded_output = padded_output[padded.mask]
    max_abs_error = float((valid_padded_output - packed_output).abs().max())
    return LayoutParityResult(
        padded_shape=tuple(padded_output.shape),
        packed_shape=tuple(packed_output.shape),
        max_abs_error=max_abs_error,
    )


def main() -> None:
    """Print the layout shapes and their valid-token parity measurement."""
    result = run_example()
    print(f"padded output: {result.padded_shape}")
    print(f"packed output: {result.packed_shape}")
    print(f"maximum absolute error: {result.max_abs_error:.3e}")


if __name__ == "__main__":
    main()
