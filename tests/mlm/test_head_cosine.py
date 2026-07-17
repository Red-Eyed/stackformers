from __future__ import annotations

import torch

from stackformers.mlm.head_cosine import CosineHead

M, D = 12, 16


def test_cosine_head_output_is_scalar(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    head = CosineHead(dim=D).to(device=device, dtype=dtype)
    prediction_in = torch.randn(M, D, device=device, dtype=dtype)
    target = torch.randn(M, D, device=device, dtype=dtype)
    loss = head(prediction_in, target)
    assert loss.shape == ()


def test_cosine_head_zero_when_prediction_matches_target_direction(device: torch.device) -> None:
    head = CosineHead(dim=D).to(device)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(D, device=device))
        head.proj.bias.zero_()
    x = torch.randn(M, D, device=device)
    loss = head(x, x)  # proj is identity, so prediction == target exactly
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_cosine_head_is_scale_invariant(device: torch.device) -> None:
    """Unlike RegressionHead's MSE, cosine similarity scores direction only — a target
    that's rescaled (e.g. by drift over training) must not change the loss at all.
    """
    head = CosineHead(dim=D).to(device)
    prediction_in = torch.randn(M, D, device=device)
    target = torch.randn(M, D, device=device)
    loss_at_original_scale = head(prediction_in, target)
    loss_at_100x_scale = head(prediction_in, target * 100.0)
    assert torch.allclose(loss_at_original_scale, loss_at_100x_scale, atol=1e-5)


def test_cosine_head_gradients_flow(device: torch.device) -> None:
    head = CosineHead(dim=D).to(device)
    prediction_in = torch.randn(M, D, device=device, requires_grad=True)
    target = torch.randn(M, D, device=device)
    loss = head(prediction_in, target)
    loss.backward()
    assert prediction_in.grad is not None
    assert head.proj.weight.grad is not None
