from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from stackformers.attention.config import DistanceBiasConfig, SelfAttentionConfig
from stackformers.attention.distance_bias import RelativeDistanceBias
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE2DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.sequence import PaddedInput
from tests.conftest import atol

B, N, D, H, DH = 2, 12, 64, 4, 16
R_MAX, NUM_RBF = 100.0, 16


def rotated(pos: torch.Tensor, radians: float) -> torch.Tensor:
    """Rotate (..., 2) node coordinates about the origin."""
    cos, sin = math.cos(radians), math.sin(radians)
    matrix = torch.tensor([[cos, -sin], [sin, cos]], device=pos.device, dtype=pos.dtype)
    return pos @ matrix.T


@pytest.fixture
def config() -> DistanceBiasConfig:
    return DistanceBiasConfig(heads=H, r_max=R_MAX, num_rbf=NUM_RBF)


@pytest.fixture
def bias(config: DistanceBiasConfig, device: torch.device) -> RelativeDistanceBias:
    """Weights randomised — the zero init would make every invariance test pass vacuously."""
    module = RelativeDistanceBias(config).to(device=device)
    nn.init.normal_(module.to_bias.weight, std=0.5)
    return module


@pytest.fixture
def positions(device: torch.device) -> torch.Tensor:
    """Scattered 2-D node coordinates — float32 by convention, like every other position."""
    return torch.rand(B, N, 2, device=device, dtype=torch.float32) * 80.0


@pytest.fixture
def nodes(positions: torch.Tensor, device: torch.device) -> PaddedInput:
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return PaddedInput(x=x, mask=mask, abs_positions=positions)


def test_bias_shape(bias: RelativeDistanceBias, nodes: PaddedInput) -> None:
    assert bias(nodes).shape == (B, H, N, N)


def test_zero_init_is_a_no_op(config: DistanceBiasConfig, nodes: PaddedInput) -> None:
    """Training starts from content-only attention and learns the distance profile."""
    fresh = RelativeDistanceBias(config).to(device=nodes.x.device)
    assert torch.all(fresh(nodes) == 0.0)


def test_invariant_to_rotation(
    bias: RelativeDistanceBias, nodes: PaddedInput, positions: torch.Tensor
) -> None:
    """The contract this module exists for — RoPE-2D fails it (see test below)."""
    turned = nodes._replace(abs_positions=rotated(positions, math.pi / 6))
    torch.testing.assert_close(bias(nodes), bias(turned), atol=atol(torch.float32), rtol=0)


def test_invariant_to_translation(
    bias: RelativeDistanceBias, nodes: PaddedInput, positions: torch.Tensor
) -> None:
    shift = torch.tensor([37.0, -12.0], device=positions.device)
    moved = nodes._replace(abs_positions=positions + shift)
    torch.testing.assert_close(bias(nodes), bias(moved), atol=atol(torch.float32), rtol=0)


def test_equal_distances_get_equal_bias(bias: RelativeDistanceBias, device: torch.device) -> None:
    """Two neighbours 15 units away — one on an axis, one diagonal — must score identically."""
    pos = torch.tensor(
        [[[0.0, 0.0], [15.0, 0.0], [10.6066, 10.6066]]], device=device, dtype=torch.float32
    )
    x = torch.randn(1, 3, D, device=device)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    out = bias(PaddedInput(x=x, mask=mask, abs_positions=pos))
    torch.testing.assert_close(out[:, :, 0, 1], out[:, :, 0, 2], atol=atol(torch.float32), rtol=0)


def test_rope2d_is_not_rotation_invariant(positions: torch.Tensor, device: torch.device) -> None:
    """Guard on the reason this module exists: axial RoPE privileges the x and y axes."""
    rope = RotaryEmbedding2D(RoPE2DConfig(dim_head=DH, base=100)).to(device)
    q = k = torch.randn(B, H, N, DH, device=device)
    turned = rotated(positions, math.pi / 6)
    qa, ka = rope.forward_padded(q, k, positions, positions)
    qb, kb = rope.forward_padded(q, k, turned, turned)
    before, after = qa @ ka.transpose(-2, -1), qb @ kb.transpose(-2, -1)
    assert (before - after).abs().mean() > 0.1 * before.abs().mean()


def test_gradients_reach_the_profile(bias: RelativeDistanceBias, nodes: PaddedInput) -> None:
    bias(nodes).sum().backward()
    grad = bias.to_bias.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


def test_self_attention_output_is_rotation_invariant(
    config: DistanceBiasConfig, nodes: PaddedInput, positions: torch.Tensor, device: torch.device
) -> None:
    """End to end: the attention block itself, not just the bias, ignores the global frame."""
    module = RelativeDistanceBias(config).to(device)
    nn.init.normal_(module.to_bias.weight, std=0.5)
    attn = SelfAttention(
        config=SelfAttentionConfig(dim=D, heads=H, dim_head=DH),
        pos_encoding=NoPosEncoding(),
        attn_bias=module,
    ).to(device)
    turned = nodes._replace(abs_positions=rotated(positions, math.pi / 6))
    torch.testing.assert_close(attn(nodes), attn(turned), atol=atol(torch.float32), rtol=0)
