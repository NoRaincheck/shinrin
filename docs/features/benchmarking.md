# Benchmarking

Shinrin includes built-in utilities for comparing model performance across training speed, prediction speed, and model size.

## Benchmark Functions

| Function | Description |
|---|---|
| `benchmark_training()` | Measure training time |
| `benchmark_prediction()` | Measure prediction time |
| `benchmark_model_size()` | Measure model file size |
| `full_benchmark()` | Run all benchmarks |
| `print_benchmark_report()` | Print formatted results |
| `ablation_benchmark()` | Fit time + held-out quality per model variant |
| `print_ablation_report()` | Print an ablation table with before/after deltas |

## Quick Benchmark

```python
from shinrin.benchmark import (
    benchmark_training,
    benchmark_prediction,
    benchmark_model_size,
    full_benchmark,
    print_benchmark_report,
)
from shinrin import MondrianTreeRegressor, MondrianForestRegressor

models = {
    "shinrin_tree": MondrianTreeRegressor(max_depth=8),
    "shinrin_forest": MondrianForestRegressor(n_estimators=10, max_depth=8),
}

# Run full benchmark
results = full_benchmark(models, X_train, y_train, X_test)
print_benchmark_report(results)
```

## Individual Benchmarks

### Training Speed

```python
results = benchmark_training(models, X_train, y_train)
```

### Prediction Speed

```python
results = benchmark_prediction(models, X_test)
```

### Model Size

```python
results = benchmark_model_size(models)
```

## Example Output

```
Benchmark Results
=================

Model: shinrin_tree
  Training Time:   0.123s
  Prediction Time: 0.004s
  Model Size:      15.2 KB

Model: shinrin_forest
  Training Time:   1.234s
  Prediction Time: 0.038s
  Model Size:      148.5 KB
```

## BitLinear Ablation (before vs after ternary quantization)

The MLP and TabM estimators support training-aware ternary weight
quantization ("BitLinear"): latent float32 weights are trained with a
straight-through estimator while the forward pass uses their
`{-1, 0, +1} * gamma` absmean approximation. Use
`ablation_benchmark()` to quantify the before/after cost of switching
it on — fit time plus held-out quality for the full-precision baseline
and each quantized variant:

```python
from shinrin import MLPClassifier
from shinrin.benchmark import ablation_benchmark, print_ablation_report

variants = {
    "fp (baseline)": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=100, random_state=0,
    ),
    "ternary/row": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=100, random_state=0,
        quantization="ternary", quantization_granularity="per_row",
    ),
    "ternary/tensor": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=100, random_state=0,
        quantization="ternary", quantization_granularity="per_tensor",
    ),
}

results = ablation_benchmark(variants, X_train, y_train, X_test, y_test)
print_ablation_report(results)
```

Example output (synthetic regression, 4,000 train / 1,000 held-out x 20,
NumPy backend):

```
========================================================================
  Ablation Report (baseline: first row)
========================================================================
variant                     fit      Δfit   test score    Δscore
------------------------------------------------------------------------
fp (baseline)           0.128s     1.00x       0.9945    +0.0000
ternary/row             0.353s     2.77x       0.9941    -0.0005
ternary/tensor          0.296s     2.32x       0.9959    +0.0013
========================================================================
```

The same pattern works for TabM (`quantization="ternary"` on
`TabMClassifier` / `TabMRegressor`; pass `alpha=0` when quantizing) and
for any other before/after comparison of estimator variants.

Measured ablations across backends and tasks live in
[`BITLINEAR_BENCHMARK.md`](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/BITLINEAR_BENCHMARK.md);
headline: ~2–25% QAT overhead at these sizes, regression R² holds at
parity, rank-based multiclass tasks are more sensitive.

## Scripts

Standalone benchmark scripts live in `scripts/benchmarks/` with committed
results documents:

| Script | Comparison | Results doc (`scripts/benchmarks/`) |
|---|---|---|
| `bench_all.py` | All algorithms across a suite of synthetic + real datasets (fit/predict/score) | `ALL_MODELS_BENCHMARK.md`, published at [Benchmark Results](benchmark-results.md) |
| `bench_gosdt.py` | GOSDT pipeline vs scikit-learn CART (speed, accuracy, tree size, optimality gap) | `GOSDT_BENCHMARK.md` |
| `bench_corels.py` | CORELS fit times, mini-GMP vs no-GMP builds | printed |
| `bench_ordt.py` | ORDT: optimal rule-sets from decision trees — skope-rules mining + CORELS selection vs cart/skope/corels (`just bench-ordt`) | `ORDT_BENCHMARK.md` |
| `bench_mlp.py` | sklearn vs shinrin MLP backends (NumPy/Mojo/PLE) | `MLP_BENCHMARK.md` |
| `bench_bitlinear.py` | Ternary (BitLinear) QAT vs full precision, MLP/TabM both backends + TabICL PTQ inference | `BITLINEAR_BENCHMARK.md` |
| `bench_backends.py` | Rust vs Mojo backends | `BENCHMARK.md` |
| `bench_tabm.py` | TabM NumPy/Mojo/PyTorch | `TABM_BENCHMARK.md` |
| `bench_tabicl.py` | TabICL fit/predict across NumPy/torch/Mojo backends with predict throughput, plus optional PTQ ablation (`--quant-ablation`) and `batch_size x kv_cache` sweep (`--cache-sweep`) | `TABICL_BENCHMARK.md` |

Run them with `just bench-gosdt`, `just bench-mlp`, `just bench-backends`, etc.
`just bench-bitlinear` runs the ternary-quantization ablation benchmark;
`just bench-all` runs `bench_all.py`, which measures every algorithm on every
dataset in the suite and republishes the results page.

There is no `just` recipe for the TabICL benchmark; run it directly:

```bash
uv run python scripts/benchmarks/bench_tabicl.py --backend numpy --quick
uv run --extra tabicl-bench python scripts/benchmarks/bench_tabicl.py \
    --backend torch --with-upstream
uv run python scripts/benchmarks/bench_tabicl.py --backend mojo \
    --quant-ablation --cache-sweep
```

The baseline tree-model comparison against LightGBM / scikit-learn lives in
[`scripts/benchmarks/bench_baselines.py`](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/bench_baselines.py)
(results in [`BENCHMARK.md`](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/BENCHMARK.md)).
