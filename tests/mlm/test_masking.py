from __future__ import annotations

import torch

from stackformers.mlm.masking import RandomMasking
from stackformers.sequence import make_packed_input, make_padded_input

B, N, D = 4, 64, 16
NT = 20


def test_random_masking_padded_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    input = make_padded_input(x, mask)
    should_mask = RandomMasking(mask_ratio=0.5)(input)
    assert should_mask.shape == mask.shape
    assert should_mask.dtype == torch.bool


def test_random_masking_packed_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 12, 20], dtype=torch.int32, device=device)
    input = make_packed_input(x, cu, max_seqlen=12)
    should_mask = RandomMasking(mask_ratio=0.5)(input)
    assert should_mask.shape == (NT,)
    assert should_mask.dtype == torch.bool


def test_random_masking_never_masks_padding(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[:, N // 2 :] = False  # second half is padding
    input = make_padded_input(x, mask)
    should_mask = RandomMasking(mask_ratio=0.9)(input)
    assert not (should_mask & ~mask).any()


def test_random_masking_respects_ratio_roughly(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Statistical check over a large sample — not exact, but far enough from any other ratio."""
    device, dtype = device_dtype
    torch.manual_seed(0)
    x = torch.randn(1, 10_000, D, device=device, dtype=dtype)
    mask = torch.ones(1, 10_000, dtype=torch.bool, device=device)
    input = make_padded_input(x, mask)
    should_mask = RandomMasking(mask_ratio=0.3)(input)
    fraction = should_mask.float().mean().item()
    assert 0.25 < fraction < 0.35
