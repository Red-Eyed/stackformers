# positional

Positional encodings applied to Q and K tensors inside attention — not to the residual stream.

`RotaryEmbedding1D` is the standard choice. It supports YaRN context extension via `RoPE1DConfig(yarn=YaRNConfig(...))`. `RotaryEmbedding2D` handles grid inputs by splitting the head dimension into row and column components. `NoPosEncoding` is a null object for cross-attention paths that need no positional information.

The `PosEncoding` protocol covers padded sequences; `PackedPosEncoding` extends it for packed sequences. `RotaryEmbedding1D` and `NoPosEncoding` satisfy both.

## Adding a new encoding

1. Add a config class with a `kind: Literal[...]` discriminator to `config.py` and include it in the `PosEncodingConfig` union.
2. Implement the class satisfying `PosEncoding` (and optionally `PackedPosEncoding`) structurally.
3. Add a `case` branch in `factory.py::build_pos_encoding`.
