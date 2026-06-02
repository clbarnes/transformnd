# List available targets.
default:
    just --list

# Generate documentation, by default under `./doc/html`.
doc docdir='doc/html':
    rm -rf {{docdir}}
    uv run --group doc pdoc \
        --output-directory {{docdir}} \
        --no-include-undocumented \
        --docformat numpy \
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

bump level:
    test -z "$(git status --porcelain)" || ( git status && false )
    uv version --bump {{level}}
    changelog release "$(uv version --short)"
    git add .
    git commit -m "Bump to v$(uv version --short)"
    git tag -a "v$(uv version --short)" -m "Release v$(uv version --short)"
