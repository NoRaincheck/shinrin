# Benchmarks

Three benchmark scripts live here:

- **`bench_baselines.py`** — compares Shinrin against LightGBM and SGD baselines (results below).
- **`bench_backends.py`** — compares the Rust and Mojo native backends against each other (`just bench-backends`).
- **`bench_tabm.py`** — compares the TabM NumPy and Mojo trainer backends (see [TABM_BENCHMARK.md](TABM_BENCHMARK.md)).

## Baselines: LightGBM / SGD

Benchmarks compare **Shinrin** (Mondrian trees/forests) against **LightGBM** and **SGD** from scikit-learn.

**Dataset:** 5,000 samples × 20 features (regression / binary classification)

> To run benchmarks yourself: `python scripts/benchmarks/bench_baselines.py`

## Regression

| Model | Train Time | Predict Time (1k samples) |
|---|---|---|
| Shinrin Tree (depth=8) | 0.021s | 8.9ms/call |
| Shinrin Forest (n=10) | 0.20s | 89ms/call |
| LightGBM Tree (8 rounds) | 0.05s | 0.17ms/call |
| LightGBM Forest (10 rounds) | 0.06s | — |
| SGDRegressor (100 iters) | 0.003s | 0.04ms/call |
| SGDRegressor (partial_fit, 100 epochs) | 0.06s | — |

## Classification

| Model | Train Time | Predict Time (1k samples) |
|---|---|---|
| Shinrin Tree (depth=8) | 0.021s | 9.6ms/call |
| Shinrin Forest (n=10) | 0.20s | 95ms/call |
| LightGBM Tree (8 rounds) | 0.05s | 0.17ms/call |
| LightGBM Forest (10 rounds) | 0.06s | — |
| SGDClassifier (100 iters) | 0.006s | 0.06ms/call |
| SGDClassifier (partial_fit, 100 epochs) | 0.05s | — |

## Notes

- **Training:** Shinrin tree training is competitive with LightGBM for single trees. Forest training is slower due to Python-level tree construction (Rust optimization pending).
- **Prediction:** LightGBM and SGD are significantly faster at prediction. Shinrin prediction runs in pure Python — Rust-backed prediction is planned.
- **Partial Fit:** SGD supports online/incremental learning via `partial_fit`. Shinrin does not yet support partial fit — this is a planned feature.
- **Shinrin strengths:** TreeSHAP explanations, ONNX export, and Mondrian tree-specific algorithms are unique features not available in LightGBM or SGD.
