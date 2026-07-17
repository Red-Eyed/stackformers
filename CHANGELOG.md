# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/): MAJOR for breaking public API changes,
MINOR for backwards-compatible features, PATCH for bug fixes and internal changes.

## [4.3.0] — 2026-07-17

### Added

- **`MLMWrapper`** (`stackformers/mlm/`) — a masked-token-reconstruction auxiliary loss over any
  encoder satisfying `EncoderLike`, domain-agnostic about what a token represents. Takes the
  encoder at call time (`forward(input, encoder)`) rather than owning it, so nothing is
  registered as a submodule and `mlm_wrapper.parameters()` never includes the encoder's weights.
- `MLMWrapperConfig`, `MaskingStrategy`/`RandomMasking`, `ReconstructionHead`/`RegressionHead`,
  and `MLMOutput` (`out`, `mlm_loss`).

### Notes

- **`out` is always the encoder's clean, unmasked output; only `mlm_loss` ever reflects
  masking.** In training, `forward` runs the encoder twice — once clean (returned as `out`) and
  once on a separately-masked copy (used only to compute `mlm_loss`) — so the main pipeline sees
  byte-identical output whether or not this loss is being trained alongside it. In eval, only the
  clean pass runs and `mlm_loss` is a constant zero, gated on `self.training` the same way
  `nn.Dropout` and `nn.BatchNorm` already are — so callers can invoke it unconditionally in both
  modes with no `if training` branch of their own.

- **The reconstruction target is always `input.x.detach()`.** This severs the gradient path from
  the loss back to whatever produced `input.x`, removing the collapse shortcut a trainable
  tokenizer would otherwise have (drive every token toward one constant vector to make
  reconstruction trivial).

- **`RandomMasking` needs no packed-sequence boundary awareness** — each token's masking decision
  is independent of every other's, so document identity (`cu_seqlens`) never enters it. That only
  becomes necessary for a contiguous-span (blockwise) strategy, which could otherwise straddle
  two packed documents.

## [4.2.1] — 2026-07-14

### Removed

- **`RoPENDConfig.headroom`.** It was redundant with `r_max`: the ladder only ever depended on
  the product `headroom · r_max`, so the two knobs shared one degree of freedom, and anyone
  wanting more reach at the slow end could get it by raising `r_max` — the parameter that has a
  measurement procedure attached. It was also the last *tuned* number in a module whose premise
  is that the band range is measured, not tuned.

  Its default has a derivation, so it need not be a parameter at all. Attention sees *signed*
  offsets spanning `[−r_max, +r_max]`, a width of `2·r_max`; asking that the slowest band turn
  through at most half a circle across that width gives `ω_lo = π / (2·r_max)` directly. That is
  bit-identical to the old `headroom=4.0` default at every `r_max`, so callers on the default —
  which is to say every caller, since the field shipped in 4.2.0 — see no change in behaviour.

  Both ends of the ladder are now the same rule, `ω = π / scale`: a half turn over `r_min` at
  the fast end (Nyquist), and a half turn over `2·r_max` at the slow end (no wrap).

  Treated as a PATCH rather than a MAJOR bump because `headroom` existed for exactly one
  release, is not known to be set anywhere, and its removal cannot change the behaviour of a
  caller that did not set it. Note that pydantic's default `extra="ignore"` means a leftover
  `RoPENDConfig(headroom=...)` is *dropped silently* rather than raising — a caller who had set
  it to something other than 4.0 will now get different frequencies without being told.

### Fixed

- **`RoPENDConfig` now requires `dim_head >= 4 * coords`** (at least two bands per axis).
  `dim_head == 2 * coords` divides cleanly and passed validation, but leaves a one-band ladder
  with nowhere to descend: `torch.linspace(hi, lo, 1)` returns `[hi]`, so the lone band lands on
  the fast end, `r_max` is discarded entirely, and the encoding becomes periodic with period
  `2 · r_min` across the whole domain — offsets of `0`, `2·r_min`, `4·r_min` … all produce an
  identical attention logit. It failed silently; it is now rejected at construction.

## [4.2.0] — 2026-07-14

### Added

- **`RotaryEmbeddingND`** (`positional/rope_nd.py`) — rotary position encoding over `c` spatial
  dimensions, splitting `dim_head` into `c` per-axis blocks. `c=1` and `c=2` reduce to the
  existing 1-D and 2-D encodings; 3-D and beyond now work with the same class.
- `RoPENDConfig`, parameterised by **`r_min` / `r_max` instead of `base`**.

### Notes

- **`base` is wrong for continuous coordinates**, and `RoPENDConfig` therefore does not have
  it. RoPE's fastest band is pinned at `ω = 1` — a wavelength of exactly 2π *coordinate units* —
  whatever the base; `base` can only stretch the slow end. So the ladder lands correctly only
  when tokens happen to sit one unit apart. That is true of text and of a patch grid, and of
  nothing else: `base=10000` on a 14×14 patch grid leaves 14 of its 16 bands frozen, and
  coordinates normalised to `[0, 1]` leave every band longer than the whole domain.

  The ladder is built from the data instead — `ω_hi = π / r_min` (Nyquist on the finest
  separation the model must resolve) down to `ω_lo = 2π / (headroom · r_max)` (the longest
  wavelength spans the domain). It then depends only on the *dynamic range* `r_max / r_min`, so
  metres, pixels and millimetres give the identical encoding — the property `base` never had,
  and the one that makes the setting measurable from the data rather than tuned.

- `RoPE1DConfig` and `RoPE2DConfig` keep `base` and are unchanged. Use them for text and grids.

- **Centre your coordinates.** Not for invariance — RoPE is translation-invariant exactly, since
  the rotations cancel into `ω·(pᵢ − pⱼ)`. For float32: the angle is `ω·p`, and coordinates far
  from the origin push it into the hundreds of radians and spend the mantissa before the cosine
  is taken. A 1e5 offset costs three orders of magnitude of accuracy.

- **It encodes direction, not just distance**, so it is not rotation invariant — by design, as
  the relative offset is strictly more information than the distance. Where the global frame is
  arbitrary, train with rotation augmentation. Exact rotation invariance in the architecture
  costs either an O(n²) bias (`RelativeDistanceBias`, above) or a canonicalisation of the input
  frame, which is inherently discontinuous: a near-isotropic point set flips frames under a 1%
  perturbation, measured on 1–2% of samples.

## [4.1.0] — 2026-07-14

### Added

- **`RelativeDistanceBias`** (`attention/distance_bias.py`) — an additive attention bias read
  off the Euclidean distance between node positions: pairwise distance → Gaussian radial
  shells → a learned per-head profile. Only `‖pᵢ − pⱼ‖` enters the logit, so attention is
  invariant to any global translation *or rotation* of the node set. For 2-D/N-D node sets
  whose coordinate frame is arbitrary.
- `DistanceBiasConfig`, `NoAttnBiasConfig`, and the `AttnBiasConfig` discriminated union;
  `attention/factory.py` with `build_attn_bias`; a `node_encoder_config` preset pairing the
  bias with `NoPosEncoding`.

### Notes

- **A rotary encoding cannot express this.** RoPE rotates by `ω·p`, which is linear in position
  by construction — and that linearity is exactly what makes the query and key rotations cancel
  into a relative offset. Distance is not linear in position, so it can only enter as a bias.
  Concretely, RoPE-2D scores two neighbours *the same distance away* 2.7× differently depending
  on whether the offset is axis-aligned or diagonal, and rotating a node set by 30° moves its
  attention logits by 83%.
- **Opt-in, and it has a ceiling.** `NoAttnBias` remains the default and returns `None`, which
  keeps `varlen_attn` available, so the bias costs nothing when unused. When used it costs a
  great deal: a bias has no varlen slot, so it forces padded SDPA and an O(n²) tensor, and the
  `(b, n, s, num_rbf)` shell intermediate — retained for backward, and `num_rbf/h` times larger
  than the bias itself — dominates activation memory. Measured OOM past roughly 2 000 nodes on
  16 GiB with 4 layers. Where that does not fit, `RotaryEmbeddingND` plus rotation augmentation
  is the cheaper trade.

## [4.0.1] — 2026-07-14

### Fixed

- **Causal and windowed attention crashed on CUDA in half precision.** `window_mask` built its
  mask at `torch.float` while `padding_mask` used the query's dtype; adding them promoted the
  sum to float32, and CUDA's SDPA rejects a mask whose dtype differs from the query
  (`RuntimeError: invalid dtype for bias`). Every causal or sliding-window call in
  `float16`/`bfloat16` was affected — that is, the configuration a model actually trains in.
  `window_mask` now takes an explicit `dtype` and `padded_sdpa` passes `q.dtype`.

  It went unnoticed because CPU's SDPA math backend tolerates the mismatch, and the suite had
  only ever run on CPU — half precision exists solely in the CUDA half of the device matrix.
  Running the untouched tree on a GPU fails 30 tests that pass locally.

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
