"""Empty reconstruction selections remain finite and preserve backward connectivity."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers.mlm.config import MLMWrapperConfig
from stackformers.mlm.head import RegressionHead
from stackformers.mlm.head_cosine import CosineHead
from stackformers.mlm.wrapper import MLMWrapper
from stackformers.sequence import SequenceInput, make_padded_input, padded_to_packed


class NoMasking(nn.Module):
    """Select no tokens without relying on a particular random draw."""

    @override
    def forward(self, input: SequenceInput) -> Tensor:
        """Return an empty selection for either sequence layout."""
        return torch.zeros(input.x.shape[:-1], dtype=torch.bool, device=input.x.device)


class TokenEncoder(nn.Module):
    """Provide a trainable encoder with no attention-specific empty-input behavior."""

    def __init__(self) -> None:
        """Register a token-wise projection for gradient checks."""
        super().__init__()
        self.proj = nn.Linear(4, 4)

    @override
    def forward(self, input: SequenceInput) -> Tensor:
        """Encode each token independently in either layout."""
        output: Tensor = self.proj(input.x)
        return output


@pytest.fixture(params=[CosineHead, RegressionHead], ids=["cosine", "regression"])
def head(request: pytest.FixtureRequest) -> CosineHead | RegressionHead:
    """Exercise the same empty-selection contract for both built-in heads."""
    result = request.param(4)
    assert isinstance(result, (CosineHead, RegressionHead))
    return result


def test_empty_head_has_zero_gradients(head: CosineHead | RegressionHead) -> None:
    """An empty mean contributes zero while retaining input and parameter gradients."""
    encoded = torch.empty(0, 4, requires_grad=True)
    loss = head(encoded, torch.empty(0, 4))
    assert loss.shape == ()
    assert loss.item() == 0.0
    loss.backward()
    assert encoded.grad is not None
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
@pytest.mark.parametrize("valid_tokens", [0, 1])
def test_empty_wrapper_selection(
    head: CosineHead | RegressionHead, packed: bool, valid_tokens: int
) -> None:
    """No selected tokens produces zero loss and does not train upstream features."""
    features = torch.randn(1, 1, 4, requires_grad=True)
    padded = make_padded_input(features, torch.full((1, 1), bool(valid_tokens)))
    input = padded_to_packed(padded) if packed else padded
    encoder = TokenEncoder()
    wrapper = MLMWrapper(MLMWrapperConfig(dim=4), masking_strategy=NoMasking(), head=head)

    result = wrapper(input, encoder)

    torch.testing.assert_close(result.out, encoder(input))
    assert result.mlm_loss.item() == 0.0
    result.mlm_loss.backward()
    assert features.grad is None
    for parameter in (*wrapper.parameters(), *encoder.parameters()):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_default_masking_can_select_nothing() -> None:
    """The default Bernoulli strategy handles its ordinary zero-selection outcome."""
    input = make_padded_input(torch.ones(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))
    wrapper = MLMWrapper(MLMWrapperConfig(dim=4))
    encoder = TokenEncoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        loss = wrapper(input, encoder).mlm_loss
    assert loss.item() == 0.0
    loss.backward()
