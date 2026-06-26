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
    mkdir -p {{docdir}}/examples
    uv run --group examples marimo export html examples/tutorial.py -o {{docdir}}/examples/tutorial.html
    uv run --group examples marimo export html examples/image.py -o {{docdir}}/examples/image.html

# Run linters and type checkers.
lint:
    uv run --group lint ruff check src tests examples bench
    uv run --group lint mypy src tests bench
    uv run --group lint ruff format --check src tests examples bench
    uv run --group examples marimo check --strict --ignore-scripts examples/*.py
    uv run --group lint pydoclint src

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

example-edit example:
    uv run --group examples marimo edit examples/{{example}}.py

example-test:
    uv run --group examples marimo export session examples --force-overwrite

# Run benchmarks.
bench:
    uv run --group test pytest --benchmark-only

bump level:
    test -z "$(git status --porcelain)" || ( git status && false )
    uv version --bump {{level}}
    changelog release "$(uv version --short)"
    git add .
    git commit -m "Bump to v$(uv version --short)"
    git tag -a "v$(uv version --short)" -m "$(changelog entry latest)"

pre-commit:
    uv run --group dev prek run --all-files

repl:
    uv run --all-groups --all-extras --with ipython ipython
