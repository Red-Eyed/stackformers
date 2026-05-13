# positional

Positional encoding modules. Applied inside attention to Q and K tensors — not to the residual stream.

## Files

| File | Contents |
|------|----------|
| `config.py` | `YaRNConfig`, `RoPE1DConfig`, `NoPosEncodingConfig`; discriminated union `PosEncodingConfig` |
| `protocols.py` | `PosEncoding` (padded), `PackedPosEncoding` (packed) |
| `rope1d.py` | `RotaryEmbedding1D` — standard RoPE for 1-D sequences; supports YaRN context extension |
| `rope2d.py` | `RotaryEmbedding2D` — RoPE for 2-D grids (row + col position ids) |
| `none.py` | `NoPosEncoding` — null object; passes Q and K through unchanged |
| `factory.py` | `build_pos_encoding(config: PosEncodingConfig) -> PosEncoding` — dispatches on `kind` |

## How it fits into attention

`SelfAttention` and `CrossAttention` accept a `PosEncoding` and call `pos_encoding.forward(q, k)` after the Q/K projections and before the kernel. The encoding modifies Q and K in-place-equivalent fashion without touching V or the residual.

## YaRN context extension

`RoPE1DConfig(dim_head=..., yarn=YaRNConfig(scale=4.0, original_max_seq_len=512))` scales the effective context window by `scale` using frequency interpolation. The base RoPE frequencies are unchanged; only the angle magnitudes are adjusted.

## Adding a new encoding

1. Add its config class to `config.py` with a `kind: Literal["your_kind"]` field and include it in the `PosEncodingConfig` union.
2. Implement the class; satisfy `PosEncoding` (and optionally `PackedPosEncoding`) structurally.
3. Add a `case NewConfig()` branch in `factory.py::build_pos_encoding`.
