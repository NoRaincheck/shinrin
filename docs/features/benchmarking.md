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

## Scripts

Standalone benchmark scripts live in `scripts/benchmarks/` with committed
results documents:

| Script | Comparison | Results doc (`scripts/benchmarks/`) |
|---|---|---|
| `bench_gosdt.py` | GOSDT pipeline vs scikit-learn CART (speed, accuracy, tree size, optimality gap) | `GOSDT_BENCHMARK.md` |
| `bench_corels.py` | CORELS fit times, mini-GMP vs no-GMP builds | printed |
| `bench_mlp.py` | sklearn vs shinrin MLP backends (NumPy/Mojo/PLE) | `MLP_BENCHMARK.md` |
| `bench_backends.py` | Rust vs Mojo backends | `BENCHMARK.md` |
| `bench_tabm.py` | TabM NumPy/Mojo/PyTorch | `TABM_BENCHMARK.md` |

Run them with `just bench-gosdt`, `just bench-mlp`, `just bench-backends`, etc.
