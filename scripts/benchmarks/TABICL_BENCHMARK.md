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
  `predict_from_representations()`, plus native KV caching via
  `build_cache()` / `predict_with_cache()`. Numeric parity against the
  torch reference holds at fp32 noise level and is pinned by the opt-in
  test suite (`SHINRIN_TABICL_PARITY_MOJO=1 uv run pytest
  tests/test_tabicl_parity.py`). The hot GEMM / softmax / GELU paths are
  SIMD-vectorized and both the large GEMMs and the per-head attention
  loop are multithreaded (pthreads); LayerNorm / RoPE / qassmax loops are
  still single-threaded scalar code (see below).
- **upstream** — optional like-for-like comparison against pip `tabicl`;
  install with `uv sync --extra tabicl-bench` and run with
  `--with-upstream`.

To run:

    uv run python scripts/benchmarks/bench_tabicl.py --quick --backend numpy
    uv run python scripts/benchmarks/bench_tabicl.py --backend torch
    just build-tabicl-mojo                                       # once
    uv run python scripts/benchmarks/bench_tabicl.py --backend mojo --quant-ablation --cache-sweep

## Methodology

Three sections, all merged per backend into
`scripts/benchmarks/tabicl_results.json` (raw data behind the tables
below; `--smoke` writes a suffixed file instead):

- **Estimator sweep** (default): fit / predict / score for classification
  (5-class), regression, and one mixed categorical case per grid size —
  {300, 1000, 1000x100 features, 5000} samples; `--quick` restricts the
  grid to 300 x 10. Reports wall-clock mean ± std over a warmup run plus
  ≥3 timed repeats, plus predict throughput (ms per 1k test rows).
- **Ternary PTQ ablation** (`--quant-ablation`): fp vs ternary
  post-training quantization (per-row / per-tensor scales) of the
  classifier checkpoint on a fixed case (1500 train x 40 features, 400
  test rows, `sklearn.datasets.make_classification`, 4 classes). Reports
  timing, held-out accuracy, and the effective zero fraction induced in
  the ternarized weights (~32–34%: MLP linears + attention output
  projections only; Q/K/V projections, biases and norms stay fp).
- **Batch-size / KV-cache sweep** (`--cache-sweep`): predict time across
  `batch_size x kv_cache` combos on a fixed mid-size case (1000 train x
  20 features, 200 test rows), isolating the chunking cliff and the
  KV-cache fix. The one-time cache build cost shows up as extra `fit`
  time and amortizes after a handful of predict calls.

`fit()` covers preprocessing + ensemble view construction (+ KV-cache
build when enabled); `predict()` covers the ensemble forward passes.
Scores are non-degeneracy guards (> majority baseline); synthetic
datasets are not a substitute for public-benchmark accuracy.

`batch_size` bounds how many test rows are decoded per call. Chunks never
change results materially (each chunk attends only over the train
prefix), but fewer, larger chunks mean fewer O(n²) attention passes. All
backends benefit; with `kv_cache=True` the chunking cliff disappears
entirely (test chunks attend cached train-side K/V).

## Results — estimator sweep, Apple Silicon (M1 Max), macOS

Quick grid (`300 x 10`, i.e. 300 train rows x 10 features, 200 test rows;
`n_estimators=8`, `batch_size=200`, `kv_cache=False`, CPU only,
2026-08):

| Task | numpy fit | numpy predict | torch fit | torch predict | mojo fit | mojo predict | score (all) |
|---|---|---|---|---|---|---|---|
| classification | 0.066s | 3.996s (20.0 ms/1k) | 0.112s | 1.286s (6.4 ms/1k) | 0.063s | 7.457s (37.3 ms/1k) | 0.750 |
| regression | 0.065s | 4.002s (20.0 ms/1k) | 0.111s | 1.351s (6.8 ms/1k) | 0.057s | 7.422s (37.1 ms/1k) | 0.999 |
| mixed categorical | 0.077s | 3.955s (19.8 ms/1k) | 0.110s | 1.286s (6.4 ms/1k) | 0.058s | 7.364s (36.8 ms/1k) | 0.915 |

- All three backends produce **identical scores** on every task — expected,
  since they load the same checkpoint and implement the same graph; this is
  the estimator-level agreement signal for the sweep.
- On CPU, torch predict is ~3× faster than NumPy here (fused kernels vs
  unfused Python-level matmuls); NumPy stays the correctness reference.
- Mojo predict is ~2× slower than NumPy and ~6× slower than torch. Two
  causes, in order of impact:
  1. **No KV cache / per-chunk re-attention** *(fixed — see sweep below
     and status section)*: the estimator decodes test rows in chunks;
     each uncached call re-runs attention over the train context from
     scratch.
  2. **Partially-optimized kernels**: GEMMs, softmax and GELU are
     SIMD-vectorized and the GEMMs are multithreaded via pthreads
     (output rows split across workers, bit-exact vs serial); the per-head
     attention loop is now threaded as well, but LayerNorm, RoPE and
     qassmax elementwise loops are still single-threaded scalar code.
- With the default `batch_size=8` the same quick-grid Mojo predict would
  take roughly an order of magnitude longer without KV caching (see the
  sweep below). `kv_cache=True` removes this cliff.

## Ternary PTQ ablation (1500 x 40, 400 test rows)

Held-out accuracy (4-class task, chance = 0.25) and predict time,
`n_estimators=8`, `batch_size=200`, 2026-08. Effective zero fraction of
the ternarized weights: **32.45%** (per-row) / **34.46%** (per-tensor),
identical across backends (same quantized checkpoint).

| Variant | numpy predict | acc | torch predict | acc | mojo predict | acc |
|---|---|---|---|---|---|---|
| fp | 46.5s | 0.938 | 10.4s | 0.938 | 64.4s | 0.938 |
| ternary/row | 44.4s | 0.250 | 9.6s | 0.250 | 60.9s | 0.250 |
| ternary/tensor | 43.9s | 0.250 | 9.6s | 0.250 | 61.3s | 0.250 |

- Quantized inference is only ~4–5% faster: the ternary scheme
  dequantizes to fp values before the GEMMs, so matmul cost is unchanged;
  only weight-loading/memory traffic shrinks slightly.
- Accuracy collapses to chance level on this synthetic case. This matches
  the load-time warning: PTQ currently spares Q/K/V projections but still
  ternarizes the MLP linears and attention output projections, which this
  checkpoint does not tolerate. Treat TabICL ternary PTQ as
  experimental-only until a quantization-aware recipe exists.

## Batch-size / KV-cache sweep (1000 x 20, 200 test rows)

Predict time per call, `n_estimators=8`, classification, 2026-08. Cache
build adds a one-time ~16s (numpy) / ~4s (torch) / ~18s (mojo) to `fit`.

| Config | numpy | torch | mojo |
|---|---|---|---|
| bs=8, no cache | 119.4s | 32.6s | 225.4s |
| bs=32, no cache | 42.3s | 11.0s | 71.3s |
| bs=128, no cache | 20.9s | 4.8s | 29.5s |
| bs=8, kv_cache | 4.74s | 4.92s | 4.71s |
| bs=32, kv_cache | 2.67s | 1.87s | 3.53s |
| bs=128, kv_cache | 2.19s | 1.08s | 3.11s |

- Without caching, shrinking `batch_size` from 128 to 8 costs **~6–8×**
  more attention work on every backend (the chunking cliff).
- With the KV cache the batch-size dependence nearly vanishes (test
  chunks attend cached train-side K/V): at `bs=8` the cache is worth
  **25×** on NumPy, **6.6×** on torch and **48×** on mojo.
- Cached mojo predict is now on par with (or faster than) uncached NumPy
  at every batch size — the native KV path makes the experimental
  backend practically usable despite its slower uncached kernels.

## Mojo backend status

Implemented natively and covered by parity tests
(`tests/test_tabicl_parity.py`, gated by `SHINRIN_TABICL_PARITY_MOJO=1`):

- Full staged graph: `stage_col` (feature grouping, SkippableLinear,
  interleaved RoPE, dual SSMax attention blocks, sentinel restore),
  `stage_row` (CLS-token aggregation into `icl_dim` representations),
  `predict_from_representations` (ICL attention + decoder head).
- Native KV cache: `build_cache(X_train, y)` captures per-block col-attn2
  K/V (per feature position, induced-vector keys) and ICL K/V (train-row
  keys) in one pass; `predict_with_cache(X_test, cache)` skips attn1 and
  the train-side re-attention entirely. Cached outputs are **bit-exact**
  vs the plain path (identical kernels on identical K/V; pinned in
  `test_mojo_kv_cache_parity`) and ~13× faster per repeated predict call
  (2000 train rows, 200-row batches: ~4.47s → ~0.34s per call on M1 Max),
  which removes the `batch_size` cliff described above.
- Head-threaded attention: the per-head score/softmax/AV loop inside each
  attention block partitions heads across pthreads, bit-exact vs serial
  (~1.9× faster predict at 2000 train rows).
- Estimator API: `representations()`, `predict_from_representations()`,
  `forward()`, `build_cache()`, `predict_with_cache()`; classification
  softmax and regression quantile decoding.
- Parameter buffer packed/validated against `shinrin._tabicl._mojo_layout`
  (length + exact offset fingerprint checked at construction, so layout
  drift fails loudly).
- Many-class hierarchical prediction (`num_classes > max_classes`): the
  estimator transparently falls back to the torch (or numpy) backend at
  `fit()` time with a warning; a kernel-level `NotImplementedError`
  remains as a safety net for direct API users.

Open work, roughly in impact order:

- Thread the remaining elementwise passes (LayerNorm, qassmax) — the
  GEMM/softmax/GELU core and the per-head attention loop are already
  SIMD + pthread-parallel.
- GEMM panel packing/blocking; persistent thread pool (kernels currently
  pthread create/join per call).
- Sanitizer coverage: blocked by the Mojo toolchain — `--sanitize
  address/thread` instrumentation compiles, but neither `mojo build` nor
  the `mojo run` JIT can resolve the sanitizer runtime symbols (the
  toolchain bundles no ASAN/TSAN runtime, and JIT sessions do not resolve
  against `DYLD_INSERT_LIBRARIES`-loaded runtimes). Revisit when the
  toolchain ships a linkable runtime.
