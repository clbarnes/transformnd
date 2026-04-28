# List available targets.
default:
    just --list

# Generate documentation under `./doc/`.
doc:
    rm -rf doc/zarr_n5
    uv run pdoc \
        --output-directory doc \
        --no-include-undocumented \
        --docformat markdown \
        --search \
        transformnd

# Run linters and type checkers.
lint:
    uv run ruff check src tests examples bench
    uv run mypy src tests bench
    uv run ruff format --check src tests examples bench

# Auto-fix format and lints where possible.
fix:
    uv run ruff check --fix src tests examples bench
    uv run ruff format src tests examples bench

# Format python code.
format:
    uv run ruff format src tests examples bench

# Run unit tests.
test:
    uv run pytest -v

# Run benchmarks.
bench:
    uv run pytest --benchmark-only
