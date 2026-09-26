"""Train one step of a small decoder-only causal language model."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from stackformers import TransformerEncoder, make_padded_input, plain_encoder_config

DIM = 64
VOCAB_SIZE = 32


class LanguageModelResult(NamedTuple):
    """Training measurements produced by :func:`run_example`."""

    logits_shape: tuple[int, ...]
    loss: float
    gradient_norm: float
    parameter_update_norm: float


class CausalLanguageModel(nn.Module):
    """Add token embeddings and a vocabulary projection around a causal encoder."""

    def __init__(self) -> None:
        """Build the embedding, causal Transformer backbone, and untied output head."""
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, DIM)
        self.backbone = TransformerEncoder(
            plain_encoder_config(dim=DIM, heads=1, num_layers=2, causal=True, ff_mult=3.0)
        )
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    @override
    def forward(self, token_ids: Tensor, mask: Tensor) -> Tensor:
        """Return next-token logits while preventing attention to future positions."""
        embeddings = self.token_embedding(token_ids)
        hidden = self.backbone(make_padded_input(embeddings, mask))
        output: Tensor = self.lm_head(hidden)
        return output

    if TYPE_CHECKING:
        __call__ = forward


def _next_token_loss(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Score each non-final position against the token immediately after it."""
    predictions = logits[:, :-1].reshape(-1, VOCAB_SIZE)
    targets = token_ids[:, 1:].reshape(-1)
    return F.cross_entropy(predictions, targets)


def run_example() -> LanguageModelResult:
    """Run a complete forward, backward, and optimizer step on token IDs."""
    torch.manual_seed(0)
    model = CausalLanguageModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_ids = torch.tensor(
        [[1, 5, 9, 2, 7, 3], [4, 8, 6, 2, 1, 0]],
        dtype=torch.long,
    )
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    embedding_before = model.token_embedding.weight.detach().clone()

    logits = model(token_ids, mask)
    loss = _next_token_loss(logits, token_ids)
    loss.backward()
    embedding_gradient = model.token_embedding.weight.grad
    if embedding_gradient is None:
        raise RuntimeError("backward did not reach the token embeddings")
    gradient_norm = float(embedding_gradient.norm())
    optimizer.step()
    parameter_update_norm = float((model.token_embedding.weight.detach() - embedding_before).norm())

    return LanguageModelResult(
        logits_shape=tuple(logits.shape),
        loss=float(loss.detach()),
        gradient_norm=gradient_norm,
        parameter_update_norm=parameter_update_norm,
    )


def main() -> None:
    """Print the causal language-model training measurements."""
    result = run_example()
    print(f"logits: {result.logits_shape}")
    print(f"loss: {result.loss:.4f}")
    print(f"embedding gradient norm: {result.gradient_norm:.3e}")
    print(f"embedding update norm: {result.parameter_update_norm:.3e}")


if __name__ == "__main__":
    main()
