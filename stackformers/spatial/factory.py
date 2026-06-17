from __future__ import annotations

from stackformers.norm.factory import build_norm
from stackformers.positional.protocols import PosEncoding
from stackformers.spatial.config import (
    ConvKVReductionConfig,
    KVReductionConfig,
    NoKVReductionConfig,
    PatchMergingConfig,
    SpatialAttnConfig,
    SpatialReductionAttentionConfig,
    WindowAttention2DConfig,
)
from stackformers.spatial.kv_reduction import ConvKVReduction, NoKVReduction
from stackformers.spatial.patch_merging import PatchMerging
from stackformers.spatial.protocols import KVReduction, SpatialAttn
from stackformers.spatial.spatial_reduction import SpatialReductionAttention
from stackformers.spatial.window import WindowAttention2D


def build_kv_reduction(config: KVReductionConfig) -> KVReduction:
    match config:
        case NoKVReductionConfig():
            return NoKVReduction()
        case ConvKVReductionConfig():
            return ConvKVReduction(config, build_norm(config.norm))
        case _:
            raise AssertionError(f"Unhandled KV reduction config: {type(config)}")


def _kv_reduction_for(config: SpatialReductionAttentionConfig) -> KVReductionConfig:
    if config.reduction == 1:
        return NoKVReductionConfig()
    return ConvKVReductionConfig(dim=config.dim, reduction=config.reduction, norm=config.kv_norm)


def build_spatial_attn(config: SpatialAttnConfig, pos_encoding: PosEncoding) -> SpatialAttn:
    match config:
        case WindowAttention2DConfig():
            return WindowAttention2D(config, pos_encoding)
        case SpatialReductionAttentionConfig():
            reducer = build_kv_reduction(_kv_reduction_for(config))
            return SpatialReductionAttention(config, pos_encoding, reducer)
        case _:
            raise AssertionError(f"Unhandled spatial attention config: {type(config)}")


def build_patch_merging(config: PatchMergingConfig) -> PatchMerging:
    return PatchMerging(config, build_norm(config.norm))
