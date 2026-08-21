# Justfile for shinrin
# Requires: just, rustfmt, cargo, uv, ruff, ty (mojo optional for the Mojo backend)

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

# ---------- Mojo backend ----------

# Build the Mojo native extension (shinrin._native_mojo_core)
build-mojo:
    uv run mojo build src/shinrin/_native_mojo.mojo --emit shared-lib -o src/shinrin/_native_mojo_core.so

# Run the test suite against the Mojo backend
test-mojo: build-mojo
    SHINRIN_BACKEND=mojo uv run pytest tests/

# Build the TabM Mojo trainer extension (shinrin._native_tabm)
build-tabm-mojo:
    uv run mojo build src/shinrin/_tabm_kernels.mojo --emit shared-lib -o src/shinrin/_native_tabm.so

# Run TabM tests against the Mojo backend
test-tabm-mojo: build-tabm-mojo
    SHINRIN_TABM_BACKEND=mojo uv run pytest tests/test_tabm_parity.py tests/test_tabm.py

# Benchmark Rust vs Mojo backends
bench-backends: build-mojo
    uv run python scripts/benchmarks/bench_backends.py

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