# AGENTS.md

- This project uses **uv** for Python package management and **maturin** for building Rust extensions.
- Keep `__init__.py` files as minimal as possible.

## Formatting & Linting

Before committing any changes, always run the formatter and linter:

```bash
just format   # Format Rust (rustfmt) and Python (ruff format)
just lint     # Lint Rust (clippy), Python (ruff check, ty)
```

If there are fixable lint issues, apply them with:

```bash
just fix      # Auto-fix Rust (clippy --fix) and Python (ruff check --fix)
```

All commits must pass `just lint` with zero warnings or errors. Do not commit code that fails lint checks.
