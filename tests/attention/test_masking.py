"""Admission and precedence tests for per-call attention visibility."""

from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from stackformers.attention.masking import SelfAttentionMask, resolve_self_mask, valid_pair_mask
from stackformers.result import Err, Ok


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("default", (True, 2, None)),
        ("causal", (True, None, None)),
        ("bidirectional", (False, None, None)),
    ],
)
def test_mode_precedence(
    mode: Literal["default", "causal", "bidirectional"], expected: tuple[bool, int | None, None]
) -> None:
    """Explicit modes replace both configured causal and window restrictions."""
    result = resolve_self_mask(mode, True, 2, 3, torch.device("cpu"))
    assert isinstance(result, Ok)
    assert result.value == expected


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 3, 3)])
def test_pair_mask_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    """Only the shared query/key matrix shape belongs to the public contract."""
    result = valid_pair_mask(torch.ones(shape, dtype=torch.bool), 3, 3, torch.device("cpu"))
    assert isinstance(result, Err)
    assert isinstance(result.error, ValueError)


def test_pair_mask_rejects_float_values() -> None:
    """Numeric masks cannot silently invert or change boolean visibility semantics."""
    result = valid_pair_mask(torch.ones(3, 3), 3, 3, torch.device("cpu"))
    assert isinstance(result, Err)
    assert isinstance(result.error, TypeError)


def test_pair_mask_rejects_wrong_device() -> None:
    """Admission does not implicitly copy a potentially large mask between devices."""
    result = valid_pair_mask(
        torch.ones(3, 3, device="meta", dtype=torch.bool), 3, 3, torch.device("cpu")
    )
    assert isinstance(result, Err)
    assert isinstance(result.error, ValueError)


def test_pair_mask_admission_preserves_identity() -> None:
    """Validation keeps the caller's tensor without allocating another mask."""
    mask = torch.eye(3, dtype=torch.bool)
    result = valid_pair_mask(mask, 3, 3, mask.device)
    assert isinstance(result, Ok)
    assert result.value is mask


def test_unknown_self_mask_is_rejected() -> None:
    """Misspelled runtime modes fail instead of changing visibility silently."""
    result = resolve_self_mask(
        cast("SelfAttentionMask", "casual"), False, None, 3, torch.device("cpu")
    )
    assert isinstance(result, Err)
    assert isinstance(result.error, ValueError)
