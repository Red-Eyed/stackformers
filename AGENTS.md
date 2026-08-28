# Repository guidance

## Python structured records

Never use `dataclasses` in this repository. PyTorch recognizes `NamedTuple` and `TypedDict`
reliably across pytree, export, and compilation boundaries, while dataclasses require additional
registration and handling.

- Use `NamedTuple` for fixed-shape records with named fields, especially tensor inputs and outputs.
- Use `TypedDict` when the value must retain mapping semantics.
- Do not introduce `dataclass`, `@dataclass`, or imports from `dataclasses`.

Ruff enforces the import restriction through `TID251`.
