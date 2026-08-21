# TabM Benchmarks

Comparison of the two TabM trainer backends shipped with shinrin:

- **numpy** — pure NumPy reference implementation (BLAS-accelerated via
  Accelerate on macOS).
- **mojo** — self-contained Mojo training kernels (`shinrin._native_tabm`,
  built with `just build-tabm-mojo`; no BLAS dependency, single-threaded).

To run: `uv run python scripts/benchmarks/bench_tabm.py [--samples N]
[--max-iter N] [--with-torch]`. The optional PyTorch reference
(`--with-torch`) requires the `tabm-bench` dependency group.

## Setup

- Default architecture: `k=32`, `hidden_layer_sizes=(256,)`, Adam,
  piecewise-linear embeddings (`n_bins=64`), quantile/asinh/scaler
  preprocessing.
- Datasets are synthetic; "mixed categorical" rounds half the columns to
  ~6 levels so they flow through the categorical one-hot path.
- Machine: Apple Silicon (M-series), macOS. Numbers are wall-clock
  `fit()` times including preprocessing.

## Results — 1,000 samples x 20 features, 100 epochs

| Task | NumPy fit | Mojo fit | NumPy score | Mojo score |
|---|---|---|---|---|
| Regression | 25.3s | 26.5s | 0.997 | 0.997 |
| 3-class | 44.7s | **29.7s** | 0.787 | 0.770 |
| Mixed categorical (binary) | **30.6s** | 39.2s | 0.986 | 0.987 |

## Results — 5,000 samples x 20 features, 100 epochs

| Task | NumPy fit | Mojo fit | NumPy score | Mojo score |
|---|---|---|---|---|
| Regression | 127.2s | 133.6s | 0.999 | 0.999 |
| 3-class | 219.4s | **153.2s** | 0.498 | 0.489 |
| Mixed categorical (binary) | **152.3s** | 278.3s | 0.966 | 0.967 |

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
