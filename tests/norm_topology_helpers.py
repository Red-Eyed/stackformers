"""Deterministic test modules for observing normalization operation order."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers.sequence import PaddedInput, SequenceInput, make_padded_input


class ScaleSelfAttention(nn.Module):
    """Multiply query embeddings while ignoring sequence metadata."""

    def __init__(self, scale: float) -> None:
        """Initialize the trainable query scale."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    @override
    def forward(self, input: SequenceInput) -> Tensor:
        """Scale the embedding tensor carried by the sequence input."""
        return input.x * self.scale


class ScaleCrossAttention(nn.Module):
    """Combine scaled queries with a context summary of matching feature width."""

    def __init__(self, query_scale: float, context_scale: float = 1.0) -> None:
        """Initialize independent trainable query and context scales."""
        super().__init__()
        self.query_scale = nn.Parameter(torch.tensor(query_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))

    @override
    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        """Add the mean context feature vector to every scaled query token."""
        context = ctx_input.x.mean(dim=-2, keepdim=True)
        return x_input.x * self.query_scale + context * self.context_scale


class ScaleFeedForward(nn.Module):
    """Multiply token embeddings with a trainable scalar."""

    def __init__(self, scale: float) -> None:
        """Initialize the scalar multiplier."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Scale every token feature uniformly."""
        return x * self.scale


class AffineNorm(nn.Module):
    """Stand-in norm whose affine transform makes operation order observable."""

    def __init__(self, scale: float, offset: float) -> None:
        """Initialize the trainable scale and offset."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.offset = nn.Parameter(torch.tensor(offset))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the configured affine transform to every feature."""
        return x * self.scale + self.offset


def make_topology_input(start: int = 0, *, requires_grad: bool = False) -> PaddedInput:
    """Build a deterministic three-token input with four features per token."""
    x = torch.arange(start, start + 12, dtype=torch.float32).reshape(1, 3, 4)
    x.requires_grad_(requires_grad)
    return make_padded_input(x, torch.ones(1, 3, dtype=torch.bool))


def expected_norm(x: Tensor) -> Tensor:
    """Apply the fixed affine equation used by topology expectations."""
    return 5.0 * x + 1.0


def expected_cross_attention(x: Tensor, context: Tensor) -> Tensor:
    """Apply the fixed cross-attention equation used by topology expectations."""
    context_summary = context.mean(dim=-2, keepdim=True)
    return 3.0 * x + context_summary
