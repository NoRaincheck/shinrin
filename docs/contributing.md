# Contributing

We welcome contributions to Shinrin! This document outlines the process for contributing.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/NoRaincheck/shinrin.git
cd shinrin
```

2. Install development dependencies:

```bash
uv sync --all-extras
```

3. Install the `just` command runner for convenient tasks.

## Project Commands

The project uses [just](https://just.systems/) for task automation. Common commands:

```bash
just format   # Format Rust (rustfmt) and Python (ruff format)
just lint     # Lint Rust (clippy), Python (ruff check, ty)
just fix      # Auto-fix linting issues
just test     # Run all tests
just check    # Run lint and test
```

## Commit Messages

All commits follow [Conventional Commits](https://www.conventionalcommits.org/) semantics:

```
feat(parser): add support for typed arrays
fix(api): resolve null pointer in request handler
docs(readme): update installation instructions
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`feature/my-feature`)
3. Make your changes
4. Run `just check` to ensure lint and tests pass
5. Submit a pull request

## Code Style

- Rust: Follow `rustfmt` and `clippy` conventions
- Python: Follow `ruff` conventions
- All type hints should be included
