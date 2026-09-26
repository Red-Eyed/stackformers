"""Customize an encoder through a focused hook or a fully user-owned preset subclass."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Self

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from torch import Tensor
from typing_extensions import override

from stackformers import (
    FeedForward,
    RoPE1DConfig,
    RotaryEmbedding1D,
    SelfAttention,
    SelfAttentionConfig,
    TransformerEncoder,
    TransformerEncoderConfig,
    TransformerLayer,
    TransformerLayerBase,
    make_padded_input,
    plain_encoder_config,
)
from stackformers.presets.encoder import TransformerEncoderBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stackformers.norm.protocols import Norm

DIM = 64


class SubclassingResult(NamedTuple):
    """Output shapes and custom-component counts produced by :func:`run_example`."""

    focused_output_shape: tuple[int, ...]
    custom_output_shape: tuple[int, ...]
    focused_custom_ff_layers: int
    custom_custom_ff_layers: int


class UserFeedForward(nn.Module):
    """User-defined gated feed-forward branch satisfying the structural protocol."""

    def __init__(self, dim: int) -> None:
        """Build bias-free gate, value, and output projections."""
        super().__init__()
        self.gate = nn.Linear(dim, dim * 2, bias=False)
        self.value = nn.Linear(dim, dim * 2, bias=False)
        self.output = nn.Linear(dim * 2, dim, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Transform tokens without changing their leading dimensions or width."""
        output: Tensor = self.output(torch.sigmoid(self.gate(x)) * self.value(x))
        return output

    if TYPE_CHECKING:
        __call__ = forward


class FocusedCustomEncoder(TransformerEncoder):
    """Replace only the feed-forward collaborator of the standard encoder preset."""

    @override
    def build_ff(self, config: TransformerEncoderConfig) -> FeedForward:
        """Inject the user feed-forward module while retaining every other preset choice."""
        return UserFeedForward(config.attn.dim)


class UserEncoderConfig(BaseModel):
    """Minimal configuration owned entirely by the user-defined preset."""

    dim: int = Field(gt=0)
    heads: int = Field(gt=0)
    num_layers: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_head_width(self) -> Self:
        """Reject geometries that cannot split the model width evenly across heads."""
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        return self


class FullyCustomEncoder(TransformerEncoderBase[UserEncoderConfig]):
    """Build every layer and final norm from a user-owned configuration."""

    def build_layers(self, config: UserEncoderConfig) -> Sequence[TransformerLayerBase]:
        """Construct pre-norm layers with the user's feed-forward implementation."""
        dim_head = config.dim // config.heads
        attention_config = SelfAttentionConfig(
            dim=config.dim,
            heads=config.heads,
            dim_head=dim_head,
        )
        positions = RotaryEmbedding1D(RoPE1DConfig(dim_head=dim_head))
        return [
            TransformerLayer(
                self_attn=SelfAttention(attention_config, pos_encoding=positions),
                ff=UserFeedForward(config.dim),
                norm_attn=nn.RMSNorm(config.dim),
                norm_ff=nn.RMSNorm(config.dim),
            )
            for _ in range(config.num_layers)
        ]

    def build_norm(self, config: UserEncoderConfig) -> Norm:
        """Construct the final normalization required by the base preset."""
        return nn.RMSNorm(config.dim)


def _custom_feed_forward_count(model: nn.Module) -> int:
    """Count user feed-forward modules to prove the override is active in every layer."""
    return sum(isinstance(module, UserFeedForward) for module in model.modules())


def run_example() -> SubclassingResult:
    """Execute both subclassing styles over the same padded token batch."""
    torch.manual_seed(0)
    focused = FocusedCustomEncoder(
        plain_encoder_config(dim=DIM, heads=1, num_layers=2, ff_mult=3.0)
    ).eval()
    custom = FullyCustomEncoder(UserEncoderConfig(dim=DIM, heads=1, num_layers=2)).eval()
    input = make_padded_input(
        torch.randn(2, 5, DIM),
        torch.ones(2, 5, dtype=torch.bool),
    )
    with torch.no_grad():
        focused_output = focused(input)
        custom_output = custom(input)
    return SubclassingResult(
        focused_output_shape=tuple(focused_output.shape),
        custom_output_shape=tuple(custom_output.shape),
        focused_custom_ff_layers=_custom_feed_forward_count(focused),
        custom_custom_ff_layers=_custom_feed_forward_count(custom),
    )


def main() -> None:
    """Print the outputs and injected-component counts for both subclassing styles."""
    result = run_example()
    print(f"focused override output: {result.focused_output_shape}")
    print(f"fully custom output: {result.custom_output_shape}")
    print(f"focused custom feed-forward layers: {result.focused_custom_ff_layers}")
    print(f"fully custom feed-forward layers: {result.custom_custom_ff_layers}")


if __name__ == "__main__":
    main()
