# TabICL Benchmarks

Comparison harness for the TabICL inference backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS). Default fallback backend.
- **torch** — own PyTorch implementation of the architecture, loading the
  same converted weights. Select with `SHINRIN_TABICL_BACKEND=torch` or
  `TabICLClassifier(backend="torch")`; `--torch` in the harness. Supports
  `device=` for GPU inference.
- **mojo** — *not benchmarked yet*. The native kernels
  (`shinrin/_tabicl_kernels.mojo`, built with `just build-tabicl-mojo`)
  compile and expose an end-to-end forward pass, but they are an
  experimental scaffold: SSMax MLPs, target-aware column masking and KV
  caching are simplified, so numeric parity with the reference backends is
  **not yet achieved** (the parity suite only smoke-tests shapes when
  explicitly enabled via `SHINRIN_TABICL_PARITY_MOJO=1`). Benchmark numbers
  will be added once the kernel reaches parity.
- **upstream** — optional like-for-like comparison against pip `tabicl`;
  install with `uv sync --extra tabicl-bench` and run with `--with-upstream`.

To run:

    uv run python scripts/benchmarks/bench_tabicl.py            # numpy sweep
    uv run python scripts/benchmarks/bench_tabicl.py --quick    # small grid
    uv run python scripts/benchmarks/bench_tabicl.py --torch    # torch sweep
    uv run python scripts/benchmarks/bench_tabicl.py --kv-cache # build caches in fit

## Methodology

- Grid: {300, 1000, 1000x100 features, 5000} samples; classification
  (5-class), regression, and one mixed categorical case per size.
- `fit()` time covers preprocessing + ensemble view construction (+ KV-cache
  build when enabled); `predict()` covers the ensemble forward passes over
  200–1000 test rows only. Warmup run then ≥3 timed repeats reporting
  mean ± std wall-clock.
- Scores are reported as non-degeneracy guards (> majority baseline);
  synthetic datasets are not a substitute for public-benchmark accuracy.

## Results — Apple Silicon (M-series), macOS, NumPy backend

First-run reference numbers (`n_estimators=8`, `batch_size=8`,
`kv_cache=False`, 300 test rows):

| Dataset | Task | fit | predict | score |
|---|---|---|---|---|
| 300 x 10 | classification | 0.06s | ~25s | 0.75 |
| 300 x 10 | regression | 0.06s | ~25s | 0.999 |
| 300 x 10 | mixed categorical | 0.06s | ~25s | 0.92 |

The NumPy backend is a correctness reference: its predict cost grows with
O(n²) attention over train rows per ensemble member and is dominated by
unfused Python-level matmuls. The torch backend (SDPA, optional GPU) is the
intended production path; Mojo kernels are tracked as future work (see
above).
