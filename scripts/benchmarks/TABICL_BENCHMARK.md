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
`n_estimators=8`, `batch_size=200`, `kv_cache=False`, CPU only;
2026-08, mojo kernels after the SIMD/register-tile work below):

| Task | numpy fit | numpy predict | torch fit | torch predict | mojo fit | mojo predict | score (all) |
|---|---|---|---|---|---|---|---|
| classification | 0.066s | 3.996s (20.0 ms/1k) | 0.112s | 1.286s (6.4 ms/1k) | 0.058s | 3.597s (18.0 ms/1k) | 0.750 |
| regression | 0.065s | 4.002s (20.0 ms/1k) | 0.111s | 1.351s (6.8 ms/1k) | 0.056s | 3.664s (18.3 ms/1k) | 0.999 |
| mixed categorical | 0.077s | 3.955s (19.8 ms/1k) | 0.110s | 1.286s (6.4 ms/1k) | 0.056s | 3.622s (18.1 ms/1k) | 0.915 |

- All three backends produce **identical scores** on every task — expected,
  since they load the same checkpoint and implement the same graph; this is
  the estimator-level agreement signal for the sweep.
- On CPU, torch predict remains ~3× faster than both reference backends
  (fused SDPA-style kernels vs unfused matmul chains).
- Mojo predict used to be ~2× slower than NumPy here; after vectorizing
  the elementwise attention paths, register-tiling `gemm_nt` / `gemm_nn`,
  pooling the worker threads and threading SSMax over rows (see kernel
  history below) it is now slightly **faster** than NumPy (~52% faster
  than its pre-optimization numbers). Remaining known gaps: LayerNorm/
  RoPE stay scalar by design (parity contracts pin exact accumulation
  orders), no fused online-softmax attention yet, GEMM panel packing/
  blocking still open.

## Ternary PTQ ablation (1500 x 40, 400 test rows)

Held-out accuracy (4-class task, chance = 0.25) and predict time,
`n_estimators=8`, `batch_size=200`, 2026-08. Effective zero fraction of
the ternarized weights: **32.45%** (per-row) / **34.46%** (per-tensor),
identical across backends (same quantized checkpoint).

| Variant | numpy predict | acc | torch predict | acc | mojo predict | acc |
|---|---|---|---|---|---|---|
| fp | 46.5s | 0.938 | 10.4s | 0.938 | 34.0s | 0.938 |
| ternary/row | 44.4s | 0.250 | 9.6s | 0.250 | 34.1s | 0.250 |
| ternary/tensor | 43.9s | 0.250 | 9.6s | 0.250 | 34.1s | 0.250 |

- Quantized inference runs at essentially the same speed as fp: the
  ternary scheme dequantizes to fp values before the GEMMs, so matmul
  cost is unchanged; only weight-loading/memory traffic shrinks slightly.
- Accuracy collapses to chance level on this synthetic case. This matches
  the load-time warning: PTQ currently spares Q/K/V projections but still
  ternarizes the MLP linears and attention output projections, which this
  checkpoint does not tolerate. Treat TabICL ternary PTQ as
  experimental-only until a quantization-aware recipe exists.

## Batch-size / KV-cache sweep (1000 x 20, 200 test rows)

Predict time per call, `n_estimators=8`, classification, 2026-08 (mojo
column after the kernel work below). Cache build adds a one-time ~16s
(numpy) / ~4s (torch) / ~10s (mojo) to `fit`.

| Config | numpy | torch | mojo |
|---|---|---|---|
| bs=8, no cache | 119.4s | 32.6s | 86.8s |
| bs=32, no cache | 42.3s | 11.0s | 30.7s |
| bs=128, no cache | 20.9s | 4.8s | 14.9s |
| bs=8, kv_cache | 4.74s | 4.92s | 3.12s |
| bs=32, kv_cache | 2.67s | 1.87s | 2.07s |
| bs=128, kv_cache | 2.19s | 1.08s | 1.69s |

- Without caching, shrinking `batch_size` from 128 to 8 costs **~6–8×**
  more attention work on every backend (the chunking cliff).
- With the KV cache the batch-size dependence nearly vanishes (test
  chunks attend cached train-side K/V): at `bs=8` the cache is worth
  **25×** on NumPy, **6.6×** on torch and **28×** on mojo.
- Cached mojo is faster than cached numpy at every batch size and within
  ~1.6× of torch — the native KV path makes the experimental backend
  practically usable.

## Mojo kernel performance history

Self-time attribution comes from `scripts/benchmarks/profile_tabicl_mojo.py`
(macOS `sample`, workload 1000 train x 40 features, 300 test rows,
`n_estimators=8`). Baseline compute split: `gemm_nt_rows` ~60%,
`ssmax_apply` ~15%, threaded head attention ~13%, split/merge/residual
glue ~9%; only ~17% of thread-time was compute (rest parked in per-region
pthread create/join cycles).

Landed (predict on that workload, M1 Max):

- **Elementwise SIMD pass**: vectorized `_split_heads`/`_merge_heads`
  chunk copies, bias adds, residual adds, SSMax scaling and qassmax dots;
  new `_dot`/`_scale_inplace`/`_add_f32` helpers. All bit-exact vs the
  prior kernels (parity suite pins staged outputs at atol 5e-5 and
  plain-vs-cached KV equality exactly). 42.3s → 31.5s (-26%).
- **GEMM register tiles**: `gemm_nt_rows` full-block path now computes
  4x2 output tiles (eight independent FMA chains instead of four;
  accumulation order per element unchanged). A 4x4 variant spilled
  registers and regressed, so it was rejected. 31.5s → 25.3s (-20%).
- **gemm_nn dual column blocks** plus a partition-bound fix (workers no
  longer write past their row range when chunk sizes are not multiples
  of four). 25.3s → 25.0s.
- **Scoped persistent worker pool**: one pool per exported entry-point
  call; workers park on a condition variable between parallel regions
  instead of pthread create/join per region. Determinism pinned by a
  stress suite (`tests/test_tabicl_pool_stress.py`: bit-equality across
  repeats, odd shapes, tiny batches, repeated model creation).
  Wall-clock ~neutral on this workload (25.0s → 25.5s) — the parked
  thread-time in profiles was mostly join-wait, not recoverable spawn
  cost — but it removes thousands of spawns per predict and provides the
  worker-scratch infrastructure for the next item.
- **SSMax row threading**: QASSMax kinds (4/5) partition rows across the
  pool with per-partition scratch; per-row math unchanged so threaded and
  serial paths stay bit-identical. 25.5s → 23.1s (-10%).

Cumulative: predict on this workload went **42.3s → 23.1s (-45%)**, and
the quick-grid estimator sweep flipped from mojo being ~2× slower than
NumPy to slightly faster (7.46s → 3.60s per call).

Open work, roughly in impact order:

- GEMM panel packing/blocking for large problems.
- Fused online-softmax attention (SDPA-style), removing the materialized
  score matrix and the split/merge transposes.
- Thread LayerNorm/RoPE over rows (tiny shares; only worth it as part of
  larger changes).
- Sanitizer coverage: blocked by the Mojo toolchain — `--sanitize
  address/thread` instrumentation compiles, but neither `mojo build` nor
  the `mojo run` JIT can resolve the sanitizer runtime symbols (the
  toolchain bundles no ASAN/TSAN runtime, and JIT sessions do not resolve
  against `DYLD_INSERT_LIBRARIES`-loaded runtimes). Revisit when the
  toolchain ships a linkable runtime.

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
  `test_mojo_kv_cache_parity`) and ~13× faster per repeated predict call,
  which removes the `batch_size` cliff described above.
- Head-threaded attention: the per-head score/softmax/AV loop inside each
  attention block partitions heads across pthreads, bit-exact vs serial.
- SIMD elementwise paths and register-tiled GEMMs (see kernel performance
  history above), all bit-exact vs their pre-optimization kernels.
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

Open work is listed at the end of the kernel performance history section
above.
