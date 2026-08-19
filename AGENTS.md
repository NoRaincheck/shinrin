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

## Commit Messages

All commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/) semantics.

Use the format: `<type>(<scope>): <description>`

Common types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

Examples:

```
feat(parser): add support for typed arrays
fix(api): resolve null pointer in request handler
docs(readme): update installation instructions
```

Keep descriptions lowercase and imperative (e.g., "add feature" not "added feature").
