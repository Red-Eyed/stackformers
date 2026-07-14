# positional

Positional encodings applied to Q and K tensors inside attention — not to the residual stream.

`RotaryEmbedding1D` is the standard choice. It supports YaRN context extension via `RoPE1DConfig(yarn=YaRNConfig(...))`. `RotaryEmbedding2D` takes two coordinates per token (a grid's row/col, or a point's x/y — the two axes are treated symmetrically, so the order is yours to pick as long as it is consistent) and splits the head dimension between them. `NoPosEncoding` is a null object for cross-attention paths that need no positional information.

RoPE's shortest wavelength is fixed at 2π coordinate units, and `base` only stretches the long end of the band range. Coordinates must therefore live at a scale the bands can resolve: positions normalised to `[0, 1]` leave every band longer than the whole domain, so the encoding barely rotates and carries almost no signal. Scale coordinates to span a few hundred units and lower `base` accordingly (≈`(extent / 2π) ** (8/7)` for `dim_head=32`).

The `PosEncoding` protocol exposes two methods — `forward_padded` and `forward_packed` — so callers (which already know their layout) pick the right path directly without internal dispatch. All implementations share the same computation via layout-agnostic helpers, since positional encoding is a per-token operation that does not depend on sequence boundaries.

## Adding a new encoding

1. Add a config class with a `kind: Literal[...]` discriminator to `config.py` and include it in the `PosEncodingConfig` union.
2. Implement the class satisfying `PosEncoding` (and optionally `PackedPosEncoding`) structurally.
3. Add a `case` branch in `factory.py::build_pos_encoding`.
