"""Convert registered model layers between separate and packed projection layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, assert_never

from stackformers.attention.config import (
    CrossAttentionConfig,
    PackedCrossAttentionConfig,
    PackedSelfAttentionConfig,
    SelfAttentionConfig,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.feedforward.config import (
    GEGLUConfig,
    HardSwishGLUConfig,
    PackedGEGLUConfig,
    PackedHardSwishGLUConfig,
    PackedSwiGLUConfig,
    SwiGLUConfig,
)
from stackformers.feedforward.geglu import GEGLU
from stackformers.feedforward.hardswish_glu import HardSwishGLU
from stackformers.feedforward.packed_geglu import PackedGEGLU
from stackformers.feedforward.packed_hardswish_glu import PackedHardSwishGLU
from stackformers.feedforward.packed_swiglu import PackedSwiGLU
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.presets.cross_attender import CrossAttender, CrossAttenderConfig
from stackformers.presets.decoder import TransformerDecoder, TransformerDecoderConfig
from stackformers.presets.encoder import TransformerEncoder, TransformerEncoderConfig
from stackformers.presets.variable_width_encoder import (
    VariableWidthEncoderLayerConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import torch.nn as nn
    from pydantic import BaseModel


_PACKED_CONFIGS: Mapping[type[BaseModel], type[BaseModel]] = {
    SelfAttentionConfig: PackedSelfAttentionConfig,
    CrossAttentionConfig: PackedCrossAttentionConfig,
    SwiGLUConfig: PackedSwiGLUConfig,
    GEGLUConfig: PackedGEGLUConfig,
    HardSwishGLUConfig: PackedHardSwishGLUConfig,
}
_UNPACKED_CONFIGS = {packed: separate for separate, packed in _PACKED_CONFIGS.items()}
_PresetConfig = TypeVar(
    "_PresetConfig",
    TransformerEncoderConfig,
    TransformerDecoderConfig,
    CrossAttenderConfig,
    VariableWidthTransformerEncoderConfig,
    VariableWidthEncoderLayerConfig,
)


def _pack_layer(module: nn.Module) -> nn.Module:
    """Convert built-in concrete layers without erasing subclass behavior."""
    match module:
        case SelfAttention() if type(module) is SelfAttention:
            return module.to_packed()
        case CrossAttention() if type(module) is CrossAttention:
            return module.to_packed()
        case SwiGLU() if type(module) is SwiGLU:
            return module.to_packed()
        case GEGLU() if type(module) is GEGLU:
            return module.to_packed()
        case HardSwishGLU() if type(module) is HardSwishGLU:
            return module.to_packed()
        case _:
            return module


def _unpack_layer(module: nn.Module) -> nn.Module:
    """Restore original concrete layers while leaving other modules in place."""
    match module:
        case PackedSelfAttention() if type(module) is PackedSelfAttention:
            return module.unpack()
        case PackedCrossAttention() if type(module) is PackedCrossAttention:
            return module.unpack()
        case PackedSwiGLU() if type(module) is PackedSwiGLU:
            return module.unpack()
        case PackedGEGLU() if type(module) is PackedGEGLU:
            return module.unpack()
        case PackedHardSwishGLU() if type(module) is PackedHardSwishGLU:
            return module.unpack()
        case _:
            return module


def _config_variant(
    config: BaseModel, variants: Mapping[type[BaseModel], type[BaseModel]]
) -> BaseModel:
    """Copy a projection config into its destination variant without changing geometry."""
    target = variants.get(type(config))
    if target is None:
        return config
    return target.model_validate(config.model_dump(exclude={"kind"}))


def _preset_config(
    config: _PresetConfig, variants: Mapping[type[BaseModel], type[BaseModel]]
) -> _PresetConfig:
    """Copy built-in preset metadata so its JSON reconstructs the converted layout."""
    match config:
        case TransformerEncoderConfig() | CrossAttenderConfig() | VariableWidthEncoderLayerConfig():
            return config.model_copy(
                update={
                    "attn": _config_variant(config.attn, variants),
                    "ff": _config_variant(config.ff, variants),
                }
            )
        case TransformerDecoderConfig():
            return config.model_copy(
                update={
                    "self_attn": _config_variant(config.self_attn, variants),
                    "cross_attn": _config_variant(config.cross_attn, variants),
                    "ff": _config_variant(config.ff, variants),
                }
            )
        case VariableWidthTransformerEncoderConfig():
            layers = [_preset_config(layer, variants) for layer in config.layers]
            return config.model_copy(update={"layers": layers})
        case _:
            assert_never(config)


def _preset_updates(
    modules: Iterable[nn.Module], variants: Mapping[type[BaseModel], type[BaseModel]]
) -> list[tuple[nn.Module, BaseModel]]:
    """Prepare new configs without mutating caller-owned config objects."""
    updates: list[tuple[nn.Module, BaseModel]] = []
    for module in modules:
        match module:
            case TransformerEncoder() if type(module) is TransformerEncoder:
                updates.append((module, _preset_config(module.config, variants)))
            case TransformerDecoder() if type(module) is TransformerDecoder:
                updates.append((module, _preset_config(module.config, variants)))
            case CrossAttender() if type(module) is CrossAttender:
                updates.append((module, _preset_config(module.config, variants)))
            case VariableWidthTransformerEncoder() if (
                type(module) is VariableWidthTransformerEncoder
            ):
                updates.append((module, _preset_config(module.config, variants)))
    return updates


def _replace_children(source: nn.Module, replacements: Mapping[int, nn.Module]) -> None:
    """Reconnect retained children, including shared collaborators and duplicate registrations."""
    target = replacements[id(source)]
    # named_children() deduplicates aliases; the registry retains every child slot.
    for name, child in source._modules.items():
        if child is not None and name in target._modules:
            target.add_module(name, replacements[id(child)])


def _apply_preset_updates(updates: Iterable[tuple[nn.Module, BaseModel]]) -> None:
    """Apply prepared metadata to the corresponding concrete preset types."""
    for module, config in updates:
        match module, config:
            case TransformerEncoder(), TransformerEncoderConfig():
                module.config = config
            case TransformerDecoder(), TransformerDecoderConfig():
                module.config = config
            case CrossAttender(), CrossAttenderConfig():
                module.config = config
            case VariableWidthTransformerEncoder(), VariableWidthTransformerEncoderConfig():
                module.config = config


def _convert(
    root: nn.Module,
    convert_layer: Callable[[nn.Module], nn.Module],
    variants: Mapping[type[BaseModel], type[BaseModel]],
) -> nn.Module:
    """Prepare replacements once per identity before applying any model changes."""
    modules = tuple(root.modules())
    replacements = {id(module): convert_layer(module) for module in modules}
    updates = _preset_updates(modules, variants)
    for module in modules:
        _replace_children(module, replacements)
    _apply_preset_updates(updates)
    return replacements[id(root)]


def to_packed(module: nn.Module) -> nn.Module:
    """Replace supported registered layers in place and return the possibly replaced root.

    Weights, dtype/device, training modes, trainability, and shared module references are
    preserved. Built-in preset configs are replaced with matching packed configs. Unknown
    modules and custom subclasses retain their behavior; supported children are still visited.
    Outputs agree within floating-point tolerance, rather than necessarily bit for bit.

    Convert before constructing an optimizer. Replaced projection parameters do not carry
    optimizer state, existing gradients, hooks, or cross-layer parameter aliases. Mixed
    trainability within a projection group raises ValueError. Cache wrappers retain their
    interface while their registered attention children follow the requested layout.
    These checks complete before the module tree changes. Repeated calls retain packed layers.
    """
    return _convert(module, _pack_layer, _PACKED_CONFIGS)


def to_unpacked(module: nn.Module) -> nn.Module:
    """Restore separate projection layers in place and return the possibly replaced root.

    Weights round-trip exactly through packing and unpacking. Built-in preset configs follow
    the restored layout. Ownership, optimizer, subclass, and rounding semantics match
    to_packed(); repeated calls retain ordinary layers. Always use the returned root.
    """
    return _convert(module, _unpack_layer, _UNPACKED_CONFIGS)
