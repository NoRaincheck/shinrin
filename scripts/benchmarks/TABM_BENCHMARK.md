# TabM Benchmarks

Comparison of the two TabM trainer backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS).
- **mojo** — self-contained Mojo training kernels (`shinrin._native_tabm`,
  built with `just build-tabm-mojo`; no BLAS dependency, single-threaded).

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

| Task | NumPy fit | Mojo fit | Torch fit† | NumPy score | Mojo score | Torch score† |
|---|---|---|---|---|---|---|
| Regression | 121.9s | 124.8s | **26.8s** | 0.999 | 0.999 | 0.999 |
| 3-class | 214.9s | 147.9s | **29.1s** | 0.498 | 0.489 | 0.499 |
| Mixed categorical (binary) | 154.5s | 261.0s | **33.8s** | 0.966 | 0.967 | 1.000 |

† Torch 5k numbers were measured before piecewise-linear embeddings
were added to the torch reference; expect similar fit times with the
PLE-enabled setup (the 1k rows above are post-PLE).

## Notes

- **Quality parity:** both backends train the same model; scores agree to
  within shuffle-RNG noise (~0.01). Gradient-level agreement is covered by
  `tests/test_tabm_parity.py`.
- **Raw kernel throughput:** on a full-batch forward+backward
  (2000x20, k=32, d_block=256) the Mojo kernels sustain ~20 GMAC/s vs
  ~28 GMAC/s for NumPy on Accelerate BLAS.
- **Where Mojo wins:** multiclass heads and small-to-medium batches, where
  per-minibatch Python overhead dominates the NumPy path.
- **Where NumPy wins:** wide piecewise-linear embedding workloads; the
  per-feature encoding GEMMs are small and launch-bound in Mojo, while
  BLAS amortizes them well.
- The Mojo backend's distinguishing feature is not raw speed but that the
  entire training step (shuffle, minibatching, Adam/L-BFGS, dropout) runs
  natively without returning to Python, with no BLAS requirement.
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
