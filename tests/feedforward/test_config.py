"""Feed-forward rounding follows each variant's parameter-matched hidden width."""

from typing import cast

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from stackformers import AlignmentConfig
from stackformers.feedforward.config import (
    FeedForwardConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLUConfig,
    PackedGEGLUConfig,
    PackedHardSwishGLUConfig,
    PackedSwiGLUConfig,
    ReluSquaredConfig,
    SwiGLUConfig,
)
from stackformers.feedforward.factory import build_ff

FeedForwardConfigType = (
    type[SwiGLUConfig]
    | type[PackedSwiGLUConfig]
    | type[GEGLUConfig]
    | type[PackedGEGLUConfig]
    | type[HardSwishGLUConfig]
    | type[PackedHardSwishGLUConfig]
    | type[GELUConfig]
    | type[ReluSquaredConfig]
)


@pytest.fixture(
    params=[
        SwiGLUConfig,
        PackedSwiGLUConfig,
        GEGLUConfig,
        PackedGEGLUConfig,
        HardSwishGLUConfig,
        PackedHardSwishGLUConfig,
        GELUConfig,
        ReluSquaredConfig,
    ]
)
def config_type(request: pytest.FixtureRequest) -> FeedForwardConfigType:
    """Cover every gated, non-gated, separate, and packed variant."""
    return cast("FeedForwardConfigType", request.param)


@pytest.mark.parametrize(
    ("mode", "gated", "non_gated"), [("floor", 640, 1024), ("ceil", 704, 1088)]
)
def test_hidden_width(
    config_type: FeedForwardConfigType, mode: str, gated: int, non_gated: int
) -> None:
    """Rounding follows GLU scaling and survives discriminated JSON serialization."""
    rounding = AlignmentConfig.model_validate({"multiple_of": 64, "rounding_mode": mode})
    config = config_type(dim=257, alignment=rounding)
    expected = non_gated if config.kind in ("gelu", "relu_squared") else gated
    assert config.inner_dim == expected
    restored = TypeAdapter(FeedForwardConfig).validate_json(config.model_dump_json())
    assert restored == config
    assert restored.inner_dim == expected
    module = build_ff(restored)
    assert isinstance(module, torch.nn.Module)
    assert any(
        expected in parameter.shape or 2 * expected in parameter.shape
        for parameter in module.parameters()
    )


def test_disabled_rounding(config_type: FeedForwardConfigType) -> None:
    """The default preserves the previous hidden-width formula."""
    config = config_type(dim=257)
    expected = 1028 if config.kind in ("gelu", "relu_squared") else 685
    assert config.inner_dim == expected


def test_floor_rejects_zero(config_type: FeedForwardConfigType) -> None:
    """A zero-width floor is rejected before constructing any layers."""
    with pytest.raises(ValidationError, match="positive width"):
        config_type(dim=1, alignment=AlignmentConfig(multiple_of=64, rounding_mode="floor"))


@pytest.fixture
def tokens(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    """Provide a small token batch in every supported device and dtype."""
    device, dtype = device_dtype
    return torch.randn(2, 3, 257, device=device, dtype=dtype)


@pytest.mark.parametrize("mode", ["floor", "ceil"])
def test_rounded_forward(
    config_type: FeedForwardConfigType, mode: str, tokens: torch.Tensor
) -> None:
    """Construction from rounded widths matches explicitly chosen hidden geometry."""
    rounding = AlignmentConfig.model_validate({"multiple_of": 64, "rounding_mode": mode})
    config = config_type(dim=257, alignment=rounding)
    scale = 1.0 if config.kind in ("gelu", "relu_squared") else 2 / 3
    # Add half a unit to avoid float truncation at an exact hidden-width boundary.
    reference_config = config_type(dim=257, mult=(config.inner_dim + 0.5) / (257 * scale))
    assert reference_config.inner_dim == config.inner_dim
    module = build_ff(config)
    reference = build_ff(reference_config)
    assert isinstance(module, torch.nn.Module)
    assert isinstance(reference, torch.nn.Module)
    module.to(device=tokens.device, dtype=tokens.dtype)
    reference.to(device=tokens.device, dtype=tokens.dtype)
    reference.load_state_dict(module.state_dict())
    torch.testing.assert_close(module(tokens), reference(tokens), rtol=0, atol=0)
