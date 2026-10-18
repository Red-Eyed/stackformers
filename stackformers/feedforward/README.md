# feedforward

Token-wise feed-forward sublayers behind the `FeedForward` protocol: `(x: b n d) → b n d`.

Gated variants (`SwiGLU` and `GEGLU`) use `int(dim * mult * 2/3)` hidden units so
their three projections have roughly the same parameter count as the two-projection `GELUFFN`
and `ReluSquaredFF`, which use `int(dim * mult)`.

## Packed SwiGLU experiment

`ff.to_packed()` creates an independent `PackedSwiGLU`. One `nn.Linear` projects gate
and value together, then a split restores the original SiLU, multiplication, dropout,
and output-projection order. The ordinary SwiGLU forward path remains unchanged.
No new protocol or base class is required.

Load ordinary weights before converting. `packed.unpack()` restores the ordinary class
and checkpoint names. Conversion preserves dtype/device, trainability and training modes,
and owns new parameters. Convert before creating an optimizer: optimizer state, gradients,
hooks and parameter aliases are not transferred. Gate and value projections must share
`requires_grad`. Default factory dispatch remains unchanged. Measure complete blocks before
selecting this variant for deployment.

## Adding a new feed-forward

1. Add a config class to `config.py` (or reuse `FeedForwardConfig` if parameterisation is identical).
2. Implement the class satisfying `FeedForward` structurally.
3. Add a `case` branch in `factory.py::build_ff`.
