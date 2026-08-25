# AGENTS.md

- This project uses **uv** for Python package management and **maturin** for building Rust extensions.
- Keep `__init__.py` files as minimal as possible.

## Incremental Implementation & Verification

Large implementations (new kernels, big refactors, multi-hundred-line files)
MUST be broken into small, independently verifiable steps. Whole-file rewrites
of large sources are error-prone and hard to review — do not do them.

Workflow:

1. **Plan the increments first.** Write down the ordered list of small steps
   before touching code. Each step should be one coherent change (e.g. "move
   the ONNX graph helpers to a new module", "add a criterion option to
   Mondrian trees").
2. **Prefer verbatim code motion over rewriting.** When splitting or moving
   code between files, copy it exactly; apply behavior changes as separate
   follow-up steps so a broken build is easy to attribute.
3. **Verify after every step.** The relevant fast check is:
   - Mojo: `just build-mojo` (or `uv run mojo build ...`) must succeed.
   - Python: run the narrowest applicable pytest subset.
   - Rust: `cargo check` / `just lint`.
   Never stack multiple unverified changes.
4. **Split large source files.** If a file grows past ~500–800 lines, split
   it into focused modules (e.g. math helpers / layers / driver) and import
   across them. Keep one module per concern.
5. **Make failures loud.** Cross-language contracts (buffer layouts, dims
   vectors, ABI shapes) get validated at construction time on BOTH sides, so
   drift becomes an immediate exception instead of silent corruption.
6. **Encode contracts in tests.** Shared layout/ABI specs belong in one
   Python module that both sides derive from, with tests that pin the exact
   sizes and names.

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
