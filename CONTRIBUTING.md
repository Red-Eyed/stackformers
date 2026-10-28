# Contributing

Bug reports should include the Stackformers version or commit, Python/PyTorch versions,
device and dtype, a minimal reproducer, and the expected and observed behavior.
Open reports and proposals in the [issue tracker](https://github.com/Red-Eyed/stackformers/issues).

## Development

Install uv and just, then run:

```bash
git clone https://github.com/Red-Eyed/stackformers.git
cd stackformers
just sync
just check
```

Use the [Justfile](Justfile) recipes for formatting, linting, type-checking, testing, and builds.
The same checks include library code, tests, and examples. Read [AGENTS.md](AGENTS.md) for
repository design and testing conventions.

## Proposing a component

Explain the research use case and why existing composition cannot express it. Keep each
variant focused, define its configuration next to its implementation, and extend the owning
module's factory and README. Implement protocols structurally. Include numerical and shape
coverage appropriate to the behavior, with compile/export checks when claiming those paths.

For public API changes, describe compatibility and migration. For performance claims, provide
reproducible measurements with hardware, versions, dtype, shapes, and timing methodology.
Do not add performance or accuracy claims based only on a synthetic smoke example.
