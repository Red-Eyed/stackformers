# positional

Positional encodings applied to Q and K tensors inside attention — not to the residual stream.

`RotaryEmbedding1D` is the standard choice. It supports YaRN context extension via `RoPE1DConfig(yarn=YaRNConfig(...))`. `RotaryEmbedding2D` takes two coordinates per token (a grid's row/col, or a point's x/y — the two axes are treated symmetrically, so the order is yours to pick as long as it is consistent) and splits the head dimension between them. `NoPosEncoding` is a null object for cross-attention paths that need no positional information.

## `base` is for grids. `RotaryEmbeddingND` is for continuous coordinates.

RoPE's shortest wavelength is fixed at 2π coordinate units, and `base` only stretches the long end of the band range. So `base` lands the ladder correctly only when tokens sit *one unit apart* — true for text, true for a patch grid, meaningless for scattered continuous coordinates. Positions normalised to `[0, 1]` leave every band longer than the whole domain, so the encoding barely rotates and carries almost no signal; and `base=10000` on a 14×14 patch grid leaves 14 of 16 bands frozen.

`RotaryEmbeddingND` (`RoPENDConfig`) drops `base` entirely and builds the ladder from the two numbers that actually mean something, in any number of dimensions. Both ends are the same rule — **a half turn over the scale in question, `ω = π / scale`**:

- `r_min` — the finest separation the model must resolve. Turned through π on the fastest band, so the shortest wavelength is `2·r_min`: the Nyquist limit, past which distinct offsets alias onto the same rotation. Measure it as a low percentile of the nearest-neighbour distance.
- `r_max` — the domain diameter. Attention sees *signed* offsets, spanning `[−r_max, +r_max]` — a width of `2·r_max`, not `r_max` — so the slowest band is given a half turn over that full width. It cannot wrap, which leaves it monotone in the offset: a coarse ramp the model reads as displacement. Measure it as a high percentile of the pairwise distance distribution.

There is no third knob, and no constant to tune: two measured distances fix the ladder. The resulting band range depends only on the *dynamic range* `r_max / r_min`, so metres, pixels and millimetres all give the identical encoding — the property `base` never had. `RoPE1DConfig` and `RoPE2DConfig` keep `base` and are unchanged; use them for text and grids.

`dim_head` must be at least `4 · coords`, so that every axis gets at least two bands. At one band per axis the ladder has nowhere to descend: it collapses onto the fast end, `r_max` is never reached, and the encoding silently becomes periodic with period `2·r_min` across the whole domain. `RoPENDConfig` rejects that at construction.

**Centre your coordinates.** Not for invariance — RoPE is translation-invariant exactly, since the query and key rotations cancel into `ω·(pᵢ − pⱼ)`. It is for float32: the angle is `ω·p`, and coordinates far from the origin push it into the hundreds of radians and spend the mantissa before the cosine is taken. A 1e5 offset costs three orders of magnitude of accuracy. This cannot be done inside the module — cross-attention must subtract the *same* constant from query and key positions, and the module sees them separately — so subtract the centroid at the input boundary.

**RoPE encodes direction, not just distance.** It is not rotation invariant, by design: it gives the model the relative *offset*, which is strictly more information than the distance. Where the global frame is arbitrary, train with rotation augmentation. Buying exact rotation invariance in the architecture costs either an O(n²) attention bias (see `attention/RelativeDistanceBias`, which OOMs past a few thousand nodes) or a canonicalisation of the input frame, which is inherently discontinuous — a near-isotropic point set flips frames under a 1% perturbation.

The `PosEncoding` protocol exposes two methods — `forward_padded` and `forward_packed` — so callers (which already know their layout) pick the right path directly without internal dispatch. All implementations share the same computation via layout-agnostic helpers, since positional encoding is a per-token operation that does not depend on sequence boundaries.

## Adding a new encoding

1. Add a config class with a `kind: Literal[...]` discriminator to `config.py` and include it in the `PosEncodingConfig` union.
2. Implement the class satisfying `PosEncoding` (and optionally `PackedPosEncoding`) structurally.
3. Add a `case` branch in `factory.py::build_pos_encoding`.
