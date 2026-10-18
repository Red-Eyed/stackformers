"""Owned weight conversion between separate and packed projection representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor


def _load_owned_state(
    source: nn.Module,
    target: nn.Module,
    state: dict[str, Tensor],
    trainable: dict[str, bool],
) -> None:
    """Restore weights, frozen parameters, and individual module training modes."""
    target.load_state_dict(state, assign=True)
    for name, parameter in target.named_parameters():
        parameter.requires_grad_(trainable[name])
    source_modes = {name: module.training for name, module in source.named_modules()}
    for name, module in target.named_modules():
        module.training = source_modes.get(name, source.training)


def load_packed_state(
    source: nn.Module, target: nn.Module, separate: tuple[str, ...], packed: str
) -> None:
    """Copy weights into a new layout; reject incompatible frozen projection groups.

    Parameter identities, existing gradients, optimizer state, and hooks are not transferred.
    Convert before constructing an optimizer or loading its state.
    """
    trainable = {
        name: parameter.requires_grad
        for name, parameter in source.named_parameters(remove_duplicate=False)
    }
    flags = {trainable[name] for name in separate}
    if len(flags) != 1:
        raise ValueError("packed projections must share requires_grad")
    source_state = source.state_dict()
    state = {name: value.clone() for name, value in source_state.items() if name not in separate}
    state[packed] = torch.cat([source_state[name] for name in separate], dim=0)
    for name in separate:
        del trainable[name]
    trainable[packed] = flags.pop()
    _load_owned_state(source, target, state, trainable)


def load_unpacked_state(
    source: nn.Module,
    target: nn.Module,
    packed: str,
    separate: tuple[str, ...],
) -> None:
    """Copy packed rows back to independently owned, checkpoint-compatible parameters."""
    source_state = source.state_dict()
    state = {name: value.clone() for name, value in source_state.items() if name != packed}
    widths = tuple(target.get_parameter(name).shape[0] for name in separate)
    chunks = source_state[packed].split(widths, dim=0)
    trainable = {
        name: parameter.requires_grad
        for name, parameter in source.named_parameters(remove_duplicate=False)
    }
    flag = trainable.pop(packed)
    for name, value in zip(separate, chunks, strict=True):
        state[name] = value.clone()
        trainable[name] = flag
    _load_owned_state(source, target, state, trainable)
