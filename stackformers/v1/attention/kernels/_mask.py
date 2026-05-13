from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def build_window_mask(
    n: int,
    s: int,
    window_size: int,
    causal: bool,
    device: torch.device,
) -> Float[Tensor, "1 1 n s"]:
    """Additive sliding-window mask (0 = keep, -inf = mask out).

    Causal: query i attends to keys in [i - window_size, i].
    Bidirectional: query i attends to keys in [i - half, i + half].
    """
    q_pos = torch.arange(n, device=device).unsqueeze(1)  # (n, 1)
    k_pos = torch.arange(s, device=device).unsqueeze(0)  # (1, s)
    if causal:
        allowed = (k_pos <= q_pos) & (k_pos >= q_pos - window_size)
    else:
        half = window_size // 2
        allowed = (k_pos >= q_pos - half) & (k_pos <= q_pos + half)
    mask = torch.zeros(1, 1, n, s, dtype=torch.float, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask
