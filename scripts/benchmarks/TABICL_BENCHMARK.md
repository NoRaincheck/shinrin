# TabICL Benchmarks

Comparison harness for the TabICL inference backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS). Default fallback backend.
- **torch** — own PyTorch implementation of the architecture, loading the
  same converted weights. Select with `SHINRIN_TABICL_BACKEND=torch` or
  `TabICLClassifier(backend="torch")`; `--torch` in the harness. Supports
  `device=` for GPU inference. This is the intended production path.
- **mojo** — *experimental scaffold; raw forward timed below, not a
  like-for-like backend*. The native kernels
  (`shinrin/_tabicl_kernels.mojo`, built with `just build-tabicl-mojo`)
  expose only an end-to-end `forward()` over a reduced computation graph;
  SSMax MLPs, CLS-token aggregation, KV caching and the staged
  representations API are simplified or absent, so numeric parity with the
  reference backends is **not yet achieved** (the parity suite only
  smoke-tests shapes when explicitly enabled via
  `SHINRIN_TABICL_PARITY_MOJO=1`). See the kernel benchmark section for
  timings and current limitations.
- **upstream** — optional like-for-like comparison against pip `tabicl`;
  install with `uv sync --extra tabicl-bench` and run with `--with-upstream`.

To run:

    uv run python scripts/benchmarks/bench_tabicl.py            # numpy sweep
    uv run python scripts/benchmarks/bench_tabicl.py --quick    # small grid
    uv run python scripts/benchmarks/bench_tabicl.py --torch    # torch sweep
    uv run python scripts/benchmarks/bench_tabicl.py --kv-cache # build caches in fit
    just build-tabicl-mojo                                      # once
    uv run python scripts/benchmarks/bench_tabicl.py --mojo     # kernel fwd: mojo vs torch

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
- The `--mojo` mode times raw end-to-end `forward()` passes on identical
  synthetic inputs against the torch backend (CPU and MPS where available),
  using the real classifier checkpoint weights. Each Mojo sample runs in its
  own throwaway subprocess so a crash cannot take down the sweep; survivor
  counts (`samples/attempts`) are reported alongside the timings.

## Results — estimator sweep, Apple Silicon (M1 Max), macOS

Quick-grid reference numbers (`300 x 10`, i.e. 300 train rows x 10
features, 200 test rows; `n_estimators=8`, `batch_size=8`,
`kv_cache=False`, CPU only):

| Task | numpy fit | numpy predict | torch fit | torch predict | score (both) |
|---|---|---|---|---|---|
| classification | 0.063s | 24.99s | 0.115s | 10.74s | 0.750 |
| regression | 0.061s | 24.61s | 0.117s | 11.60s | 0.999 |
| mixed categorical | 0.060s | 24.67s | 0.122s | 11.28s | 0.915 |

- Both backends produce **identical scores** on every task — expected,
  since they load the same checkpoint and implement the same graph; this is
  the estimator-level agreement signal for the sweep.
- On CPU, torch predict is ~2.1–2.3× faster than NumPy (fused kernels vs
  unfused Python-level matmuls); NumPy stays the correctness reference.
- NumPy predict cost grows with O(n²) attention over train rows per
  ensemble member; larger grid sizes are run by dropping `--quick`.

## Results — raw kernel forward: Mojo vs torch (Apple M1 Max)

Environment: Apple M1 Max, 64 GB RAM, macOS 26.6; Python 3.14.3,
PyTorch 2.13.0 (CPU + MPS), Mojo 1.0.0. Weights: real TabICLv2 classifier
checkpoint (`tabicl-classifier-v2`, embed_dim=128, icl_dim=512,
max_classes=10, ff factor 2). Input: synthetic float32 rows with 100
features; single end-to-end `forward()` call per sample
(`bench_tabicl.py --mojo`, repeat=3).

| train × test | Mojo forward¹ | torch-CPU forward | torch-MPS forward |
|---|---|---|---|
| 300 × 150 | 0.056s ±0.000 (3/3) | 0.512s ±0.023 | 0.167s ±0.001 |
| 500 × 200 | 0.072s ±0.000 (3/3) | 0.790s ±0.018 | 0.235s ±0.003 |
| 2000 × 200 | 0.156s ±0.001 (3/3) | 2.233s ±0.062 | 0.685s ±0.005 |
| 5000 × 500 | 0.363s ±0.004 (3/3) | 7.081s ±0.102 | 1.910s ±0.064 |

¹ Every Mojo sample runs in a fresh process; "n/m" counts successful samples
over attempts actually made. Wall-clock only; no numeric correctness
is asserted (see caveats).

### Analysis

- **These are throughput indicators, not like-for-like comparisons.** On the
  same inputs the two backends do very different work:
  - Mojo reads only the first feature group (`col_feature_group_size=3`) of
    each training row and never encodes test-row features (they enter as
    zero embeddings).
  - ICL aggregation collapses training rows into per-class mean embeddings
    instead of attending over all train rows, so the Mojo cost grows
    roughly linearly while the reference graph is O(n_test · n_train) in
    the attention stage.
  - SSMax MLPs, CLS-token aggregation, the trained test-row projection
    and KV caching are simplified or absent.
- **Observed speedups (~9–20× vs torch-CPU, ~3–5× vs MPS) therefore say
  "the current kernels are fast for what they compute", not "Mojo TabICL is
  X× faster than torch".**
- **Reliability:** earlier builds crashed nondeterministically (SIGSEGV in
  tcmalloc free-list handling, ~87% of isolated forwards at some sizes).
  The memory-safety fixes below resolved this: repeated current sweeps
  complete every sample (`3/3`) at all four sizes including 5000 × 500.
- **Fixed while investigating the crashes** (memory-safety hardening, no
  parity claim):
  - Python wrapper truncated the native `(n_test, out_dim)` output to
    `n_test` floats (`_tabicl/_mojo_backend.py`); shape smoke test passes.
  - Missing loop increment in the QKV bias-add scalar tail
    (`self_attn_block_forward`), which wrote without bound whenever
    `d_model % 8 != 0`.
  - All three stage workspaces sized their scratch slots at `n × embed_dim`
    although the shared FFN helper writes `n × d_ff` into one slot (a 2×
    overrun per block at ff factor 2). Slots are now sized by the stage's
    feed-forward width.
  - The flat parameter buffer is packed and validated against
    `shinrin._tabicl._mojo_layout` as the single source of truth; the
    kernel checks the packed length AND an exact offset fingerprint at
    construction, so parameter-layout drift fails loudly.
- **Open issues blocking a meaningful comparison:**
  - No sanitizer coverage: `mojo build --sanitize address` currently fails
    to link against the bundled ASAN runtime; a `mojo run`-based harness is
    the follow-up option.
  - The staged `representations` / KV-cache API needed by the estimator is
    not implemented natively yet, so Mojo cannot run the estimator sweep
    above (no fit/predict/score row).
  - Reduced graph vs torch (documented in `_tabicl_kernels.mojo`): CLS-token
    aggregation, `out_ln`/cls_tokens usage and a trained test-row projection
    are still missing, so end-to-end numeric parity remains open even though
    per-stage semantics match the reference.

Until those land, torch remains the production backend; NumPy stays the
correctness reference, and Mojo numbers should be re-taken after parity
work (they will also need the estimator-level methodology above to be
comparable).
