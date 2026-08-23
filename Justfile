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
    uv run ty check --exclude "src/shinrin/_skgarden/**" --exclude "src/shinrin/_skrules/**" --exclude "src/shinrin/_corels/**" --exclude "src/shinrin/_gosdt/**" --exclude "tests/vendored/**" src/ tests/

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

# Build the MLP Mojo trainer extension (shinrin._native_mlp)
build-mlp-mojo:
    uv run mojo build src/shinrin/_mlp_kernels.mojo --emit shared-lib -o src/shinrin/_native_mlp.so

# Run MLP tests against the Mojo backend
test-mlp-mojo: build-mlp-mojo
    SHINRIN_MLP_BACKEND=mojo uv run pytest tests/test_mlp_parity.py tests/test_mlp.py

# Benchmark sklearn vs shinrin MLP backends
bench-mlp:
    uv run python scripts/benchmarks/bench_mlp.py

# Benchmark ternary (BitLinear) quantization vs full precision
bench-bitlinear:
    uv run python scripts/benchmarks/bench_bitlinear.py

# Build the TabICL Mojo native extension (shinrin._native_tabicl)
build-tabicl-mojo:
    uv run mojo build src/shinrin/_tabicl_kernels.mojo --emit shared-lib -I src -o src/shinrin/_native_tabicl.so

# Run TabICL tests against the Mojo backend
test-tabicl-mojo: build-tabicl-mojo
    SHINRIN_TABICL_BACKEND=mojo uv run pytest tests/test_tabicl_parity.py tests/test_tabicl.py

# Benchmark GOSDT vs scikit-learn CART
bench-gosdt:
    uv run python scripts/benchmarks/bench_gosdt.py

# Benchmark Rust vs Mojo backends
bench-backends: build-mojo
    uv run python scripts/benchmarks/bench_backends.py

# Benchmark all shinrin algorithms across the dataset suite (~45 min; --smoke for a quick pass)
bench-all:
    uv run python scripts/benchmarks/bench_all.py

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