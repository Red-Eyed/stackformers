# feedforward

Token-wise feed-forward sublayers behind the `FeedForward` protocol: `(x: b n d) → b n d`.

Gated variants (`SwiGLU` and `GEGLU`) use `int(dim * mult * 2/3)` hidden units so
their three projections have roughly the same parameter count as the two-projection `GELUFFN`
and `ReluSquaredFF`, which use `int(dim * mult)`.

## Packed gated projections

`SwiGLU`, `GEGLU`, and `HardSwishGLU` expose `to_packed()` returning their corresponding
`Packed*` module. One `nn.Linear` projects gate
and value together, then a split restores the original activation, multiplication, dropout,
and output-projection order. GEGLU retains tanh-approximate GELU; HardSwishGLU retains
`nn.Hardswish`. The ordinary forward paths remain unchanged.
No new protocol or base class is required.

Load ordinary weights before converting. `packed.unpack()` restores the ordinary class
and checkpoint names. Conversion preserves dtype/device, trainability and training modes,
and owns new parameters. Convert before creating an optimizer: optimizer state, gradients,
hooks and parameter aliases are not transferred. Gate and value projections must share
`requires_grad`. Select `PackedSwiGLUConfig`, `PackedGEGLUConfig`, or
`PackedHardSwishGLUConfig` in `build_ff` or a preset to construct packed projections directly.
Their `kind` values are `packed_swiglu`, `packed_geglu`, and `packed_hardswish_glu` and survive
JSON serialization. Ordinary configs keep the original layout. Each constructor accepts only
its matching concrete config, and conversion updates the destination config accordingly. Measure complete
blocks before selecting this variant for deployment.

Each packed variant owns its activation and forward equation in a separate module. The existing
state-conversion helpers handle weight ownership and trainability; factories handle selection.
No shared GLU behavior hierarchy is needed. Sequential projections in `GELUFFN` and
`ReluSquaredFF` cannot be packed this way because the second projection consumes the activated
result of the first.

## Adding a new feed-forward

1. Add a config class to `config.py` (or reuse `FeedForwardConfig` if parameterisation is identical).
2. Implement the class satisfying `FeedForward` structurally.
3. Add a `case` branch in `factory.py::build_ff`.
