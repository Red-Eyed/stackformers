from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from stackformers.positional.config import RoPENDConfig
from stackformers.positional.rope_nd import RotaryEmbeddingND
from tests.conftest import atol

B, N, H = 2, 16, 4
R_MIN, R_MAX, HEADROOM = 0.5, 100.0, 4.0

DIMS = [(1, 64), (2, 64), (3, 96)]  # (coords, dim_head) — dim_head must divide by 2 * coords


@pytest.fixture(params=DIMS, ids=lambda p: f"c{p[0]}")
def config(request: pytest.FixtureRequest) -> RoPENDConfig:
    coords, dim_head = request.param
    return RoPENDConfig(
        dim_head=dim_head, coords=coords, r_min=R_MIN, r_max=R_MAX, headroom=HEADROOM
    )


@pytest.fixture
def rope(config: RoPENDConfig, device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbeddingND:
    device, dtype = device_dtype
    return RotaryEmbeddingND(config).to(device=device, dtype=dtype)


@pytest.fixture
def positions(config: RoPENDConfig, device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    """Scattered continuous coordinates — float32 by convention, like every other position."""
    device, _ = device_dtype
    return torch.rand(B, N, config.coords, device=device, dtype=torch.float32) * 80.0


@pytest.fixture
def qk(
    config: RoPENDConfig, device_dtype: tuple[torch.device, torch.dtype]
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    shape = (B, H, N, config.dim_head)
    return (
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
    )


def logits(rope: RotaryEmbeddingND, qk_pair: tuple[torch.Tensor, torch.Tensor], pos: torch.Tensor):
    q, k = rope.forward_padded(*qk_pair, pos, pos)
    return q @ k.transpose(-2, -1)


def test_output_shape(
    rope: RotaryEmbeddingND,
    qk: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    config: RoPENDConfig,
) -> None:
    q, k = rope.forward_padded(*qk, positions, positions)
    assert q.shape == k.shape == (B, H, N, config.dim_head)


def test_ladder_spans_nyquist_to_domain(config: RoPENDConfig) -> None:
    """The whole point of dropping `base`: the band range is pinned to the data, not a constant."""
    inv_freq: torch.Tensor = RotaryEmbeddingND(config).inv_freq  # type: ignore[assignment]
    shortest = 2 * math.pi / float(inv_freq.max())
    longest = 2 * math.pi / float(inv_freq.min())
    assert shortest == pytest.approx(2 * R_MIN, rel=1e-4)  # Nyquist on the finest separation
    assert longest == pytest.approx(HEADROOM * R_MAX, rel=1e-4)  # spans the domain, with headroom
    assert inv_freq.shape == (config.bands_per_axis,)


def test_units_do_not_matter(config: RoPENDConfig, device: torch.device) -> None:
    """Scale the coordinates and r_min/r_max together — the encoding must not move.

    This is the property `base` never had, and the reason it is gone: the ladder is a function
    of the dynamic range r_max / r_min alone, so metres, pixels and millimetres all encode
    identically. With `base`, the same geometry in different units gives a different model.
    """
    pos = torch.rand(B, N, config.coords, device=device) * 80.0
    q = k = torch.randn(B, H, N, config.dim_head, device=device)

    metres = RotaryEmbeddingND(config).to(device)
    millimetres = RotaryEmbeddingND(
        config.model_copy(update={"r_min": R_MIN * 1000, "r_max": R_MAX * 1000})
    ).to(device)

    # Identical in real arithmetic — omega scales by 1/1000 exactly as the coordinates scale by
    # 1000 — so the only gap is float32 rounding on angles of a few hundred radians. See
    # test_precision_degrades_far_from_origin for why that bound is not tighter.
    torch.testing.assert_close(
        logits(metres, (q, k), pos),
        logits(millimetres, (q, k), pos * 1000),
        atol=1e-3,
        rtol=1e-3,
    )


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().mean() / a.abs().mean()).item()


def test_invariant_to_translation(
    rope: RotaryEmbeddingND,
    qk: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Exact and free — the rotations cancel into p_i - p_j, so no centring step is needed.

    Asserted as a relative error rather than elementwise closeness: the claim is that the two
    logit *fields* agree, and in half precision an elementwise bound would be measuring the
    dtype's noise floor, not the invariance.
    """
    _, dtype = device_dtype
    shift = torch.full((positions.shape[-1],), 37.0, device=positions.device)
    moved = logits(rope, qk, positions + shift)
    assert relative_error(logits(rope, qk, positions), moved) < 10 * atol(dtype)


def test_precision_degrades_far_from_origin(device: torch.device) -> None:
    """Centre your coordinates. Not for invariance — for float32.

    Translation invariance is exact in real arithmetic, but the rotation angle is omega·p, and
    float32 carries only ~7 digits. Coordinates far from the origin push that angle into the
    hundreds or thousands of radians and spend the mantissa before the cosine is even taken.
    A 1e5 offset costs three orders of magnitude of accuracy — so subtract the centroid at the
    input boundary. It cannot be done inside this module: cross-attention would have to
    subtract the *same* constant from the query and key positions, and the module sees them
    separately.
    """
    config = RoPENDConfig(dim_head=64, coords=2, r_min=R_MIN, r_max=R_MAX)
    rope = RotaryEmbeddingND(config).to(device)
    pos = torch.rand(B, N, 2, device=device) * 80.0
    q = k = torch.randn(B, H, N, 64, device=device)
    reference = logits(rope, (q, k), pos)

    def drift(offset: float) -> float:
        moved = logits(rope, (q, k), pos + offset)
        return ((reference - moved).abs().max() / reference.abs().mean()).item()

    assert drift(37.0) < 1e-3  # centred: fine
    assert drift(1e5) > 1e-2  # far from the origin: the encoding has visibly decayed
    assert drift(1e5) > 50 * drift(37.0)


def test_preserves_norms(
    rope: RotaryEmbeddingND,
    qk: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """It is a rotation: every channel pair is turned, never stretched."""
    _, dtype = device_dtype
    q, _ = rope.forward_padded(*qk, positions, positions)
    torch.testing.assert_close(qk[0].norm(dim=-1), q.norm(dim=-1), atol=atol(dtype), rtol=1e-3)


def test_not_rotation_invariant(device: torch.device) -> None:
    """Documented limitation: it encodes direction, not just distance.

    Where the global frame is arbitrary, that has to be handled with augmentation — buying it
    in the architecture costs an O(n^2) bias or a discontinuous canonicalisation.
    """
    config = RoPENDConfig(dim_head=64, coords=2, r_min=R_MIN, r_max=R_MAX)
    rope = RotaryEmbeddingND(config).to(device)
    pos = torch.rand(B, N, 2, device=device) * 80.0
    q = k = torch.randn(B, H, N, 64, device=device)

    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    turned = pos @ torch.tensor([[c, -s], [s, c]], device=device).T

    before, after = logits(rope, (q, k), pos), logits(rope, (q, k), turned)
    assert (before - after).abs().mean() > 0.1 * before.abs().mean()


def test_packed_matches_padded(config: RoPENDConfig, device: torch.device) -> None:
    rope = RotaryEmbeddingND(config).to(device)
    pos = torch.rand(N, config.coords, device=device) * 80.0
    q = k = torch.randn(N, H, config.dim_head, device=device)

    packed_q, _ = rope.forward_packed(q, k, pos, pos)
    padded_q, _ = rope.forward_padded(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        pos.unsqueeze(0),
        pos.unsqueeze(0),
    )
    torch.testing.assert_close(
        packed_q, padded_q.squeeze(0).transpose(0, 1), atol=atol(torch.float32), rtol=0
    )


def test_gradients_flow(config: RoPENDConfig, device: torch.device) -> None:
    rope = RotaryEmbeddingND(config).to(device)
    pos = torch.rand(B, N, config.coords, device=device) * 80.0
    q = torch.randn(B, H, N, config.dim_head, device=device, requires_grad=True)
    k = torch.randn(B, H, N, config.dim_head, device=device, requires_grad=True)
    rope.forward_padded(q, k, pos, pos)[0].sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_rejects_indivisible_head_dim() -> None:
    with pytest.raises(ValidationError, match="divisible by 2 \\* coords"):
        RoPENDConfig(dim_head=64, coords=3, r_min=R_MIN, r_max=R_MAX)


def test_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError, match="must exceed r_min"):
        RoPENDConfig(dim_head=64, coords=2, r_min=100.0, r_max=1.0)
