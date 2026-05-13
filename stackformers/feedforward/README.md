# feedforward

Token-wise feed-forward sublayers behind the `FeedForward` protocol: `(x: b n d) → b n d`.

`SwiGLU` is the only implementation. Its inner dimension is `int(dim * mult * 2/3)`, scaled down so parameter count matches a standard 4× GELU FFN.

## Adding a new feed-forward

1. Add a config class to `config.py` (or reuse `FeedForwardConfig` if parameterisation is identical).
2. Implement the class satisfying `FeedForward` structurally.
3. Add a `case` branch in `factory.py::build_ff`.
