# Justfile for shinrin
# Requires: just, rustfmt, cargo, uv, ruff, ty

# ---------- Format ----------

# Format Rust and Python source files
format:
    rustfmt --edition 2021 src/lib.rs
    uv run ruff format src/ tests/

# Fix Rust and Python linting issues
fix:
    cargo clippy --fix --allow-dirty --allow-staged
    uv run ruff check --fix src/ tests/

# ---------- Lint ----------

# Lint Rust and Python code
lint:
    cargo clippy --all-targets --all-features -- -D warnings
    uv run ruff check src/ tests/
    uv run ty check --exclude "src/shinrin/_skgarden/**" --exclude "src/shinrin/_skrules/**" --exclude "tests/vendored/**" src/ tests/

# ---------- Test ----------

# Run Python and Rust tests
test:
    uv run pytest tests/
    cargo test

# Run Rust tests only
test-rust:
    cargo test

# Run Python tests only
test-python:
    uv run pytest tests/

# ---------- Docs ----------

# Build documentation locally
build-docs:
    uv run --group docs mkdocs build --strict

# Serve documentation locally
serve-docs:
    uv run --group docs mkdocs serve

# ---------- Check ----------

# Run lint and test
check: lint test