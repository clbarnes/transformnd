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
    uv run ruff check src tests
    uv run mypy src tests
    uv run ruff format --check src tests

# Format python code.
format:
    uv run ruff format src tests

# Run unit tests.
test:
    uv run pytest -v
