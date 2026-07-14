from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from stackformers.attention.config import DistanceBiasConfig
from stackformers.sequence import PaddedInput


class RelativeDistanceBias(nn.Module):
    """Additive attention bias read off the Euclidean distance between node positions.

    Only ‖p_i − p_j‖ enters the logit, so attention is invariant to any global translation
    or rotation of the node set: the encoding privileges no axis and no origin. Each head
    mixes a shared bank of Gaussian radial-basis shells into its own profile over distance,
    so some heads can go local and others long-range.

    A rotary encoding cannot express this. RoPE rotates by ω·p, which is linear in position
    by construction — that linearity is exactly what makes the query and key rotations cancel
    into a relative offset. Distance is not linear in position, so it can only enter as a bias.

    Costs one (b, h, n, s) tensor, which rules out varlen_attn (it has no bias slot) and forces
    the padded SDPA path. Affordable for node counts in the low thousands; beyond that the
    (b, n, s, num_rbf) shell intermediate, not the bias, is what dominates activation memory.
    """

    def __init__(self, config: DistanceBiasConfig) -> None:
        super().__init__()
        centres = torch.linspace(0.0, config.r_max, config.num_rbf)
        self.register_buffer("centres", centres, persistent=False)
        self.width = config.r_max / (config.num_rbf - 1)  # neighbouring shells cross at 1/e
        self.to_bias = nn.Linear(config.num_rbf, config.heads, bias=False)
        nn.init.zeros_(self.to_bias.weight)  # start at content-only attention, learn the profile

    def _shell_weights(self, dist: Float[Tensor, "b n s"]) -> Float[Tensor, "b n s k"]:
        centres: Tensor = self.centres  # type: ignore[assignment]
        return torch.exp(-(((dist.unsqueeze(-1) - centres) / self.width) ** 2))

    def forward(self, input: PaddedInput) -> Float[Tensor, "b h n s"]:
        """float32 distances keep half-precision coordinates from collapsing nearby nodes."""
        pos = input.abs_positions.float()
        delta = pos.unsqueeze(2) - pos.unsqueeze(1)  # b n s c
        dist = torch.linalg.vector_norm(delta, dim=-1)  # b n s
        shells = self._shell_weights(dist).to(self.to_bias.weight.dtype)
        bias = rearrange(self.to_bias(shells), "b n s h -> b h n s")
        return bias.to(input.x.dtype)
