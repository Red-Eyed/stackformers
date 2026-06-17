# spatial

Attention over a 2-D grid of image tokens, for efficient vision backbones. Self-contained:
nothing here touches the 1-D `SequenceInput` / `attention/` stack — it only reuses
`padded_sdpa`, position encodings, norms, and feed-forwards.

## Design

**A separate input type, not a widened `SequenceInput`.** `SpatialInput` carries the grid
shape `(H, W)` so attention can reshape the flat token sequence back to `(b, d, H, W)`. It is
deliberately kept out of the `SequenceInput` union: that union stays a sealed 1-D abstraction,
and spatial attention gets its own `SpatialAttn` protocol. Padded-only — fixed-resolution
images do not vary in length, so there is no packed spatial variant.

**Two attention variants, one protocol.** Both satisfy `SpatialAttn` and stay dot-product
attention (so RoPE 2-D, GQA, and `qk_norm` all apply):
- `WindowAttention2D` — non-overlapping `w×w` tiles, O(n·w²). Local; cheapest at the
  high-resolution early stages. Distinct from `SelfAttentionConfig.window_size`, which is a
  *1-D sliding* window over a flat sequence; this partitions the *2-D grid* into square tiles.
- `SpatialReductionAttention` — full-grid queries attend to a spatially-reduced K/V context
  (PVTv2 SRA), O(n·s) with s ≪ n. Global; affordable once the grid is small.

**Reduction is an injected `KVReduction`.** `SpatialReductionAttention` delegates K/V
downsampling to a collaborator (`NoKVReduction` for r=1 full attention, `ConvKVReduction` for a
strided conv). The reduction's norm is itself an injected `Norm` — pluggable like everywhere
else, defaulting to RMSNorm.

**Window inverse is a closure.** `partition_windows` returns `(windows, merge)`; the `merge`
inverse is bound to the exact `(b, H, W, window)` used to partition, so the un-partition step
can never be called with mismatched parameters.

**Export-friendly.** Grid shapes are Python ints throughout (positions, window counts, merged
grid sizes), so reshapes stay static under `torch.export` / `torch.compile`.

## Extending

Add a new variant as a new `nn.Module` satisfying `SpatialAttn`, plus a config with a `kind`
discriminator and a `match` arm in `factory.py`. To add a new K/V reduction, implement
`KVReduction` and extend `build_kv_reduction`. Shifted windows would be a new variant here, not
a flag on `WindowAttention2D`.

The opinionated multi-scale assembly (stages + patch-merging → feature pyramid) lives in
`presets/pyramid_vision.py`, not here — this module only provides the reusable primitives.

## References

- **Swin Transformer** — window partitioning. Liu et al., 2021. https://arxiv.org/abs/2103.14030
- **Pyramid Vision Transformer (PVT)** — spatial-reduction attention. Wang et al., 2021. https://arxiv.org/abs/2102.12122
- **PVTv2** — overlapping patch embedding / improved pyramid. Wang et al., 2022. https://arxiv.org/abs/2106.13797
