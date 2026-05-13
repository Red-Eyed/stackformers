from __future__ import annotations

import pytest
import torch

_DEVICE_DTYPE: list[tuple[str, torch.dtype]] = [
    ("cpu", torch.float32),
    ("cpu", torch.float16),
]
if torch.cuda.is_available():
    _DEVICE_DTYPE += [
        ("cuda", torch.float32),
        ("cuda", torch.float16),
        ("cuda", torch.bfloat16),
    ]

_DEVICES = list(dict.fromkeys(d for d, _ in _DEVICE_DTYPE))  # deduplicated, ordered


def _dd_id(pair: tuple[str, torch.dtype]) -> str:
    device, dtype = pair
    return f"{device}-{str(dtype).split('.')[-1]}"


@pytest.fixture(params=_DEVICE_DTYPE, ids=_dd_id)
def device_dtype(request: pytest.FixtureRequest) -> tuple[torch.device, torch.dtype]:
    """All (device, dtype) combinations available on this machine."""
    device_str, dtype = request.param
    return torch.device(device_str), dtype


@pytest.fixture(params=_DEVICES)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Devices available on this machine; float32 only — used for gradient tests."""
    return torch.device(request.param)


def atol(dtype: torch.dtype) -> float:
    """Absolute tolerance appropriate for dtype in numerical assertions."""
    return 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
