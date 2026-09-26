set dotenv-load := false
set positional-arguments

python_sources := "stackformers/ tests/ examples/"

default:
    @just --list

sync:
    uv sync --group dev

build:
    uv build --clear

fmt *args:
    uv run ruff format {{ python_sources }} "$@"

lint *args:
    uv run ruff check {{ python_sources }} "$@"

types:
    uv run pyrefly check {{ python_sources }}

test *args:
    uv run pytest -q "$@"

check: (fmt "--check") lint types test
