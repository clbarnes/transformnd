# List available targets.
default:
    just --list

# Generate documentation, by default under `./doc/`.
doc docdir='doc':
    rm -rf {{docdir}}/transformnd
    uv run --group doc pdoc \
        --output-directory {{docdir}} \
        --no-include-undocumented \
        --docformat markdown \
        --search \
        transformnd

# Run linters and type checkers.
lint:
    uv run --group lint ruff check src tests examples bench
    uv run --group lint mypy src tests bench
    uv run --group lint ruff format --check src tests examples bench

# Auto-fix format and lints where possible.
fix:
    uv run --group lint ruff check --fix src tests examples bench
    uv run --group lint ruff format src tests examples bench

# Format python code.
format:
    uv run --group lint ruff format src tests examples bench

# Run unit tests.
test:
    uv run --all-groups --all-extras pytest -v

# Run benchmarks.
bench:
    uv run --group test pytest --benchmark-only
