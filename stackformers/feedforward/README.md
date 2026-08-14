# feedforward

Token-wise feed-forward sublayers behind the `FeedForward` protocol: `(x: b n d) → b n d`.

Gated variants (`SwiGLU` and `GEGLU`) use `int(dim * mult * 2/3)` hidden units so
their three projections have roughly the same parameter count as the two-projection `GELUFFN`
and `ReluSquaredFF`, which use `int(dim * mult)`.

## Adding a new feed-forward

1. Add a config class to `config.py` (or reuse `FeedForwardConfig` if parameterisation is identical).
2. Implement the class satisfying `FeedForward` structurally.
3. Add a `case` branch in `factory.py::build_ff`.
