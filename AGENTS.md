# Repository guidance

## Repository commands

Read `Justfile` before running setup, build, formatting, linting, type-checking, testing, or
cleanup commands. Prefer its existing recipes so local and automated workflows use the same
tool arguments and defaults.

## Python structured records

Never use `dataclasses` in this repository. PyTorch recognizes `NamedTuple` and `TypedDict`
reliably across pytree, export, and compilation boundaries, while dataclasses require additional
registration and handling.

- Use `NamedTuple` for fixed-shape records with named fields, especially tensor inputs and outputs.
- Use `TypedDict` when the value must retain mapping semantics.
- Do not introduce `dataclass`, `@dataclass`, or imports from `dataclasses`.

Ruff enforces the import restriction through `TID251`.
