# Contribute

## Guidelines

- Update the documentation when you change or add public functionality.
- Follow the conventions in the repo's `CLAUDE.md` (absolute imports, `snake_case` files, `PascalCase` classes, stage classes end in `Stage`).

## Setup

```bash
git clone <your-fork-url>
cd stream_aie
git checkout -b <feature-or-fix>

# Python >= 3.12
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install        # ruff check + ruff format on every commit
```

If you use VS Code, install the **Ruff** extension (`charliermarsh.ruff`), enable format-on-save, and disable other Python formatters to avoid conflicts.

## Coding style

- **Formatter / linter** — ruff-format and `ruff check` (rules E, F, W, I, PL, N, UP, B); line length **120**.
- **Python target** — 3.12+: use `X | Y` unions and built-in generics (`list[X]`, `dict[K, V]`).
- **Imports** — absolute only; isort order stdlib → third-party → internal.
- **Type hints** — required on public functions, classes, and methods.
- **Docstrings** — Google-style.

## Before opening a PR

```bash
ruff check .
ruff format --check .
pytest tests/ -m "not slow"
```

Add or update tests for your change, update the docs if public APIs change, then open a pull request.

Thanks for contributing!
