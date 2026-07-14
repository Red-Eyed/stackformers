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
    """Absolute tolerance appropriate for dtype in numerical assertions.

    Calibrated per dtype rather than per width. bfloat16 trades mantissa for exponent, leaving
    8 explicit bits against float16's 11, so it is ~8x coarser — sharing one tolerance between
    them looked fine for as long as the suite only ran on CPU (whose matrix has no bfloat16 at
    all) and produced five failures the first time it ran on a GPU.

    float16's bound is 2e-2, not 1e-2: these assertions compare values that accumulate over
    dim_head, so the per-element error compounds. At 1e-2 the RoPE norm and relative-distance
    tests sat right on the edge and failed perhaps one run in five, on CPU and CUDA alike.
    """
    if dtype is torch.bfloat16:
        return 8e-2
    if dtype is torch.float16:
        return 2e-2
    return 1e-5
