# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/): MAJOR for breaking public API changes,
MINOR for backwards-compatible features, PATCH for bug fixes and internal changes.

## [4.0.0] — 2026-07-14

### Removed

- **Breaking: the 2-D vision modules.** The `spatial` package (`WindowAttention2D`,
  `SpatialReductionAttention`, `ConvKVReduction`/`NoKVReduction`, `PatchMerging`,
  `SpatialInput`, `SpatialTransformerLayer`) and the `PyramidVisionBackbone` preset with
  `pyramid_vision_config` are gone. The library returns to sequence models only.
- `RotaryEmbedding2D` and `RoPE2DConfig` are **kept** — they are independent of `spatial` and
  encode any two-coordinate position (grid row/col, or point x/y).

### Notes

- A coordinate-frame bug in `ConvKVReduction` (pooled keys were positioned in the reduced
  grid's index space while queries used full-grid coordinates, so a relative encoding saw an
  offset that grew with absolute position) was found but not fixed, as the module is removed
  here. Anyone reviving this code from history must fix it first: pooled tokens belong at their
  block centre, `i·r + (r − 1)/2`, in the original grid's frame.

## [3.9.2] — 2026-07-14

### Fixed

- **`RotaryEmbedding2D` was not computing a rotation.** The frequency vector was laid out as
  `[row | row | col | col]`, but `_rotate_half` pairs channel `i` with channel `i + dh/2`, so a
  row angle was paired against a column angle. The resulting 2×2 map had
  `det = cos((row − col) · ωᵢ)` and singular values `√(1 ± |sin D|)` — a squeeze, not a rotation.
  It was therefore non-orthogonal, annihilated high-frequency channels wherever the determinant
  reached zero, and produced attention scores that leaked absolute position instead of depending
  only on relative offset. Frequencies are now built as `[row | col | row | col]`.
- `RotaryEmbedding2D` now casts positions and `inv_freq` to float32 for the outer product,
  matching `RotaryEmbedding1D`. A `.half()`-ed module previously lost position precision.

### Changed

- **Numerics of `RotaryEmbedding2D` change.** Any checkpoint trained against the previous
  encoding is tuned to the broken position code and requires retraining or finetuning.

### Added

- Math tests for `RotaryEmbedding2D` on a genuine row-major grid: a definition-first reference
  oracle, orthogonality (norms and inner products), relative-offset invariance, per-axis
  sensitivity, distinct-offset separation, and packed/padded agreement. The previous test used a
  `row == col` position grid — the main diagonal, the one locus where the row and column angles
  coincide and the broken layout is indistinguishable from a correct one.

## [3.9.1] — 2026-06-30

### Fixed

- Make the experimental `varlen_attn` import and call robust across PyTorch versions.

## [3.9.0] — 2026-06-17

### Added

- 2-D grid attention (`spatial`) and the pyramid vision backbone preset.

## [3.8.2] — 2026-05-29

### Fixed

- Force absolute positions to float32 in `sequence`.

## [3.8.1] — 2026-05-29

### Fixed

- Keep batch size as a `SymInt` in the packed attention fallback, preserving `torch.export`
  compatibility.

## [3.8.0] — 2026-05-27

### Fixed

- Replace `.tolist()` loops in `sequence` and `ops` with tensor operations, making them
  `torch.export`-compatible.

### Added

- PyPI publish metadata in `pyproject.toml`.

## [3.7.0] — 2026-05-21

### Changed

- Introduce abstract generic base classes for all presets.

## [3.6.0] — 2026-05-21

### Added

- `AttnBias` protocol and the `NoAttnBias` null object on `SelfAttention`.

## [3.5.2] — 2026-05-21

### Fixed

- Warn and fall back to SDPA when `varlen_attn` raises.

## [3.5.1] — 2026-05-18

### Changed

- Move attention dropout to the output projection.

## [3.5.0] — 2026-05-18

### Added

- Packed→padded SDPA fallback; extract `ops.py`.

## [3.4.2] — 2026-05-18

### Added

- `padded_to_packed` and `packed_to_padded` helpers.

## [3.4.1] — 2026-05-18

### Removed

- Packed-attention CPU fallback; warn on dropout instead.

## [3.4.0] — 2026-05-17

### Added

- ReLU² feed-forward network.

## [3.3.0] — 2026-05-17

### Added

- QK-norm in attention; all config fields documented.

## [3.2.0] — 2026-05-17

### Changed

- Eliminate layout-specific duplicates in positional encoding implementations.

## [3.1.0] — 2026-05-17

### Changed

- Replace `forward()` on positional encodings with `forward_padded` / `forward_packed`, so
  callers pick the path for the layout they already know.

## [3.0.1] — 2026-05-17

### Fixed

- Use the GELU tanh approximation in GEGLU.

## [3.0.0] — 2026-05-16

### Changed

- **Breaking:** remove the kernel abstraction; padded and packed dispatch are unified.

## [2.0.2] — 2026-05-16

### Removed

- The unused local-attention package.

## [2.0.1] — 2026-05-16

### Changed

- Store feed-forward activations as `nn.Module` attributes.

## [2.0.0] — 2026-05-16

### Removed

- **Breaking:** the `attn_bias` / ALiBi system.

## [1.1.0] — 2026-05-16

### Added

- `LearnedPosEncoding` with absolute position embeddings.

## [1.0.1] — 2026-05-16

### Added

- `LayerNorm` wrapper.

## [1.0.0] — 2026-05-16

### Added

- GEGLU feed-forward variant.

### Changed

- **Breaking:** feed-forward config becomes a discriminated union.

## [0.5.0] — 2026-05-16

### Added

- Unfold mode for the windowed SDPA kernel.

## [0.4.0] — 2026-05-15

### Added

- `RotaryEmbedding2D` and the packed cross-attender.

## [0.3.0] — 2026-05-14

### Changed

- Establish the semantic versioning convention.

## [0.2.0] — 2026-05-13

### Changed

- Remove the `v1` namespace; add the `presets` package.

## [0.1.0] — 2026-05-13

### Added

- Initial library scaffold.
