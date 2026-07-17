from __future__ import annotations

import torch
import torch.nn as nn

from stackformers.mlm.head import RegressionHead

M, D = 12, 16


def test_regression_head_output_is_scalar(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    head = RegressionHead(dim=D).to(device=device, dtype=dtype)
    prediction_in = torch.randn(M, D, device=device, dtype=dtype)
    target = torch.randn(M, D, device=device, dtype=dtype)
    loss = head(prediction_in, target)
    assert loss.shape == ()


def test_regression_head_zero_when_projection_matches_target(device: torch.device) -> None:
    head = RegressionHead(dim=D).to(device)
    nn.init.zeros_(head.proj.weight)
    nn.init.zeros_(head.proj.bias)
    target = torch.zeros(M, D, device=device)
    prediction_in = torch.randn(M, D, device=device)  # proj(x) == 0 regardless, weight/bias are 0
    loss = head(prediction_in, target)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_regression_head_gradients_flow(device: torch.device) -> None:
    head = RegressionHead(dim=D).to(device)
    prediction_in = torch.randn(M, D, device=device, requires_grad=True)
    target = torch.randn(M, D, device=device)
    loss = head(prediction_in, target)
    loss.backward()
    assert prediction_in.grad is not None
    assert head.proj.weight.grad is not None
