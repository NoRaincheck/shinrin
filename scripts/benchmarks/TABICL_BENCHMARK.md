# TabICL Benchmarks

Comparison harness for the TabICL inference backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS). Default fallback backend and correctness reference.
- **torch** — own PyTorch implementation of the architecture, loading the
  same converted weights. Supports `device=` for GPU inference. This is the
  intended production path.
- **mojo** — *experimental*. Native Mojo inference kernels
  (`shinrin/_tabicl_kernels.mojo`, built with `just build-tabicl-mojo`)
  running the **full staged computation graph** (column embedding → row
  interaction → in-context predictor) via `representations()` /
  `predict_from_representations()`. Numeric parity against the torch
  reference holds at fp32 noise level and is pinned by the opt-in test
  suite (`SHINRIN_TABICL_PARITY_MOJO=1 uv run pytest
  tests/test_tabicl_parity.py`). Not yet competitive: the hot GEMM /
  softmax / GELU paths are SIMD-vectorized and the large GEMMs are
  multithreaded (pthreads), but there is no KV caching and the per-head
  attention / LayerNorm loops are still single-threaded scalar code
  (see below).
- **upstream** — optional like-for-like comparison against pip `tabicl`;
  install with `uv sync --extra tabicl-bench` and run with
  `--with-upstream`.

To run:

    uv run python scripts/benchmarks/bench_tabicl.py --quick --backend numpy
    uv run python scripts/benchmarks/bench_tabicl.py --backend torch
    just build-tabicl-mojo                                       # once
    uv run python scripts/benchmarks/bench_tabicl.py --backend mojo --quick

## Methodology

- Grid: {300, 1000, 1000x100 features, 5000} samples; classification
  (5-class), regression, and one mixed categorical case per size.
  (`--quick` restricts the grid to 300 x 10.)
- `fit()` time covers preprocessing + ensemble view construction (+ KV-cache
  build when enabled); `predict()` covers the ensemble forward passes over
  200–1000 test rows only. Warmup run then ≥3 timed repeats reporting
  mean ± std wall-clock.
- Scores are reported as non-degeneracy guards (> majority baseline);
  synthetic datasets are not a substitute for public-benchmark accuracy.
- `batch_size` bounds how many test rows are decoded per call. Chunks never
  change results (each chunk attends only over the train prefix), but fewer,
  larger chunks mean fewer O(n²) attention passes. All backends benefit;
  the Mojo backend has no KV cache and gains the most from large batches,
  so the sweep below pins `batch_size=200` for every backend.

## Results — estimator sweep, Apple Silicon (M1 Max), macOS

Quick grid (`300 x 10`, i.e. 300 train rows x 10 features, 200 test rows;
`n_estimators=8`, `batch_size=200`, `kv_cache=False`, CPU only):

| Task | numpy fit | numpy predict | torch fit | torch predict | mojo fit | mojo predict | score (all) |
|---|---|---|---|---|---|---|---|
| classification | 0.067s | 3.94s | 0.110s | 1.34s | 0.055s | 8.30s | 0.750 |
| regression | 0.073s | 3.95s | 0.116s | 1.38s | 0.060s | 8.39s | 0.999 |
| mixed categorical | 0.064s | 3.92s | 0.111s | 1.31s | 0.059s | 8.38s | 0.915 |

- All three backends produce **identical scores** on every task — expected,
  since they load the same checkpoint and implement the same graph; this is
  the estimator-level agreement signal for the sweep.
- On CPU, torch predict is ~3× faster than NumPy here (fused kernels vs
  unfused Python-level matmuls); NumPy stays the correctness reference.
- Mojo predict is ~2× slower than NumPy and ~6× slower than torch. Two
  causes, in order of impact:
  1. **No KV cache / per-chunk re-attention**: the estimator decodes test
     rows in chunks; each Mojo call re-runs the full O(n_train²) ICL
     attention from scratch. The torch/NumPy backends amortize this with
     `build_cache` / `predict_with_cache` when `kv_cache=True`; the Mojo
     backend raises `NotImplementedError` there yet.
  2. **Partially-optimized kernels**: GEMMs, softmax and GELU are
     SIMD-vectorized and the GEMMs are multithreaded via pthreads
     (output rows split across workers, bit-exact vs serial); but the
     per-head attention score/AV GEMMs, LayerNorm, RoPE and qassmax
     elementwise loops are still single-threaded scalar code.
- With the default `batch_size=8` the Mojo predict column would be ~85s
  instead of ~8.3s — same outputs, 25× more attention passes.

## Mojo backend status

Implemented natively and covered by parity tests
(`tests/test_tabicl_parity.py`, gated by `SHINRIN_TABICL_PARITY_MOJO=1`):

- Full staged graph: `stage_col` (feature grouping, SkippableLinear,
  interleaved RoPE, dual SSMax attention blocks, sentinel restore),
  `stage_row` (CLS-token aggregation into `icl_dim` representations),
  `predict_from_representations` (ICL attention + decoder head).
- Estimator API: `representations()`, `predict_from_representations()`,
  `forward()`; classification softmax and regression quantile decoding.
- Parameter buffer packed/validated against `shinrin._tabicl._mojo_layout`
  (length + exact offset fingerprint checked at construction, so layout
  drift fails loudly).

Open work, roughly in impact order:

- Native `build_cache` / `predict_with_cache`: cache encoded train-side
  K/V so test chunks do not re-run train attention (removes cause 1).
- Thread the per-head attention loop (scores/AV GEMMs) and the remaining
  elementwise passes (LayerNorm, qassmax) — the GEMM/softmax/GELU core is
  already SIMD + pthread-parallel (removes most of cause 2).
- Hierarchical many-class prediction (`num_classes > max_classes`) falls
  back to `NotImplementedError`; use the numpy/torch backends there.
- Sanitizer coverage: `mojo build --sanitize address` currently fails to
  link against the bundled ASAN runtime; a `mojo run`-based harness is the
  follow-up option.

Until the cache/vectorization work lands, torch remains the production
backend, NumPy stays the correctness reference, and Mojo numbers should be
re-taken after each optimization step using the estimator-level
methodology above.
