# TabM Benchmarks

Comparison of the two TabM trainer backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS).
- **mojo** — self-contained Mojo training kernels (`shinrin._native_tabm`,
  built with `just build-tabm-mojo`; no BLAS dependency). Training runs
  data-parallel across performance cores via AsyncRT task groups: one
  worker per core computes forward/backward on a private row slice of
  each minibatch, then gradients are reduced SIMD-wise. Set
  `SHINRIN_TABM_THREADS` to override the worker count (default:
  number of performance cores).

To run: `uv run python scripts/benchmarks/bench_tabm.py [--samples N]
[--max-iter N] [--with-torch]`. The optional PyTorch reference
(`--with-torch`) requires the upstream `tabm` package:

    uv pip install torch 'tabm@git+https://github.com/yandex-research/tabm.git'

The torch reference mirrors our default setup for a like-for-like
comparison — same backbone (`d_block=256`, `n_blocks=1`, `k=32`,
`arch_type='tabm'`, dropout 0.1), Adam at lr 1e-3, batch size 200,
asinh + standardization followed by piecewise-linear embeddings
(`n_bins=64`, `d_embedding=8`) — and exercises regression, multiclass
and mixed-categorical tasks.

## Setup

- Default architecture: `k=32`, `hidden_layer_sizes=(256,)`, Adam,
  piecewise-linear embeddings (`n_bins=64`), quantile/asinh/scaler
  preprocessing.
- Datasets are synthetic; "mixed categorical" rounds half the columns to
  ~6 levels so they flow through the categorical one-hot path.
- Machine: Apple Silicon (M-series), macOS. Numbers are wall-clock
  `fit()` times including preprocessing.

## Results — 1,000 samples x 20 features, 100 epochs

| Task | NumPy fit | Mojo fit | Torch fit | NumPy score | Mojo score | Torch score |
|---|---|---|---|---|---|---|
| Regression | 23.8s | 24.6s | **6.2s** | 0.997 | 0.997 | 0.995 |
| 3-class | 42.6s | 28.2s | **6.4s** | 0.787 | 0.770 | 0.880 |
| Mixed categorical (binary) | 29.8s | 36.6s | **6.9s** | 0.986 | 0.987 | 0.999 |

## Results — 5,000 samples x 20 features, 100 epochs

| Task | NumPy fit | Mojo fit | Torch fit | NumPy score | Mojo score | Torch score |
|---|---|---|---|---|---|---|
| Regression | 121.9s | 124.8s | **31.8s** | 0.999 | 0.999 | 0.998 |
| 3-class | 214.9s | 147.9s | **33.7s** | 0.498 | 0.489 | 0.784 |
| Mixed categorical (binary) | 154.5s | 261.0s | **37.0s** | 0.966 | 0.967 | 1.000 |

## Results — parallel Mojo kernels vs torch, 5,000 samples x 20 features, 20 epochs

After multithreading (`TaskGroup` row-sliced workers, one per performance
core) and SIMD-vectorizing the scalar hot loops, the Mojo backend trains
faster than the PyTorch reference on identical workloads (Apple M1 Max,
default threads):

| Task | Mojo fit | Torch fit | Mojo score | Torch score |
|---|---|---|---|---|
| Regression | **3.2–3.5s** | 6.1s | 0.997 | 0.990 |
| Binary+multiclass | **4.0–4.5s** | 6.5s | 0.435 | 0.518 |
| Mixed categorical | **5.9–6.7s** | 7.1s | 0.948 | 0.976 |

Thread scaling (regression / mixed categorical fit time):

| Threads | 1 | 2 | 8 (default) | 16 |
|---|---|---|---|---|
| Regression | 20.9s | 10.8s | 3.5s | 3.1s |
| Mixed categorical | 44.4s | 22.6s | 6.2s | 5.8s |

Scores are stable across thread counts (~±0.001); for a fixed thread
count results are fully deterministic.

## Notes

- **Quality parity:** both backends train the same model; scores agree to
  within shuffle/dropout-RNG noise (~0.01). Gradient-level agreement is
  covered by `tests/test_tabm_parity.py`. The Mojo dropout RNG draws one
  64-bit word per 8 mask decisions (byte thresholding), so trajectories
  differ from the NumPy path but keep-rate statistics are equivalent.
- **Raw kernel throughput:** on a full-batch forward+backward
  (2000x20, k=32, d_block=256) the pre-parallelization serial kernels
  sustained ~20 GMAC/s vs ~28 GMAC/s for NumPy on Accelerate BLAS; the
  current kernels add SIMD-vectorized Adam/reductions/embedding paths on
  top of the threaded GEMM fan-out.
- **Where Mojo wins:** end-to-end wall-clock on CPU — minibatch shuffle,
  dropout, member-loss reduction, Adam/L-BFGS and gradient reduction all
  run natively without returning to Python, and GEMMs scale across
  performance cores. No BLAS requirement.
- **Where NumPy wins:** nothing at these sizes any more; it remains
  useful as the readable reference implementation.
- **Torch reference (upstream `yandex-research/tabm`):** configured
  like-for-like with the NumPy/Mojo fits — same backbone shape
  (`d_block=256`, `n_blocks=1`), BatchEnsemble weight sharing
  (`arch_type='tabm'`), `k=32`, dropout 0.1, Adam at lr 1e-3,
  batch_size=200 minibatches, and proper mean-of-member losses
  (MSE / cross-entropy over the k predictions, not loss-of-mean).
  An earlier revision compared against upstream defaults
  (`d_block=512`, `n_blocks=3`, full-batch training), which made torch
  look ~2x slower than NumPy; on identical work PyTorch is in fact
  ~4-8x faster than the NumPy path here.
- Remaining caveats on the torch column: timings cover the training
  loop only (NumPy/Mojo are end-to-end `fit()` including quantile-bin
  preprocessing and scoring), it runs on CPU, and its
  hyperparameters are not tuned per task, so treat the torch scores as
  a sanity check rather than a quality claim.
