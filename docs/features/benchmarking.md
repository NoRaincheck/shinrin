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

## Ablation Benchmark (before vs after a feature)

`ablation_benchmark()` quantifies the before/after cost of toggling a
feature — fit time plus held-out quality for the baseline and each
variant. Pass the same estimator type with one option changed:

```python
from shinrin import MondrianForestRegressor
from shinrin.benchmark import ablation_benchmark, print_ablation_report

variants = {
    "small (baseline)": MondrianForestRegressor(
        n_estimators=10, random_state=0,
    ),
    "large": MondrianForestRegressor(
        n_estimators=40, random_state=0,
    ),
}

results = ablation_benchmark(variants, X_train, y_train, X_test, y_test)
print_ablation_report(results)
```

Example output:

```
========================================================================
  Ablation Report (baseline: first row)
========================================================================
variant                     fit      Δfit   test score    Δscore
------------------------------------------------------------------------
small (baseline)        0.128s     1.00x       0.9945    +0.0000
large                   0.296s     2.32x       0.9959    +0.0013
========================================================================
```

## Scripts

Standalone benchmark scripts live in `scripts/benchmarks/` with committed
results documents:

| Script | Comparison | Results doc (`scripts/benchmarks/`) |
|---|---|---|
| `bench_all.py` | All algorithms across a suite of synthetic + real datasets (fit/predict/score) | `ALL_MODELS_BENCHMARK.md`, published at [Benchmark Results](benchmark-results.md) |
| `bench_tabarena.py` | All algorithms on a core subset of the TabArena-v0.1 benchmark — 13 curated OpenML datasets spanning regression, binary and multiclass classification (`just bench-tabarena`) | `TABARENA_BENCHMARK.md` |
| `bench_gosdt.py` | GOSDT pipeline vs scikit-learn CART (speed, accuracy, tree size, optimality gap) | `GOSDT_BENCHMARK.md` |
| `bench_corels.py` | CORELS fit times, mini-GMP vs no-GMP builds | printed |
| `bench_ordt.py` | ORDT: optimal rule-sets from decision trees — skope-rules mining + CORELS selection vs cart/skope/corels (`just bench-ordt`) | `ORDT_BENCHMARK.md` |
| `bench_backends.py` | Rust vs Mojo backends | `BENCHMARK.md` |

Run them with `just bench-gosdt`, `just bench-backends`, etc.
`just bench-all` runs `bench_all.py`, which measures every algorithm on every
dataset in the suite and republishes the results page. `just bench-tabarena`
runs `bench_tabarena.py`, which measures the same algorithm matrix on a core
subset of [TabArena](https://arxiv.org/abs/2506.16791)-v0.1 (OpenML suite 457):
13 curated real-world datasets — 5 regression, 6 binary and 3 multiclass
classification — fetched from OpenML on first use. Scores are not comparable
with the public TabArena leaderboard (single fixed split, untuned budgets);
the point is comparing shinrin's algorithms against each other on curated
real-world data.

The baseline tree-model comparison against LightGBM / scikit-learn lives in
[`scripts/benchmarks/bench_baselines.py`](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/bench_baselines.py)
(results in [`BENCHMARK.md`](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/BENCHMARK.md)).
