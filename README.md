# Shinrin

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models and tabular neural networks, with Rust and Mojo bindings for performance and ONNX export support.

Since skope-rules and scikit-garden are no longer actively maintained, this project aims to bring them together with extensions for tree models — including SHAP explanations, ONNX export, and benchmarking utilities.

Shinrin also includes **TabM** — a parameter-efficient ensemble MLP for tabular data (ICLR 2025) that trains an ensemble of *k* members jointly through BatchEnsemble-style multiplicative adapters, matching ensemble accuracy at a fraction of the training cost. Training runs entirely on NumPy or Mojo kernels — PyTorch is not required.

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
- **Tabular Neural Networks** — scikit-learn compatible `MLPClassifier`/`MLPRegressor` and `TabMClassifier`/`TabMRegressor` with optional PLE embeddings and Mojo-accelerated training
- **TabM Neural Networks** — Parameter-efficient ensemble MLPs for tabular data with BatchEnsemble-style multiplicative adapters (ICLR 2025)
- **TreeSHAP Explanations** — `TreeExplainer` for single trees and forests with `explanation()` visualization helper
- **ONNX Export** — Export trained models to ONNX format for deployment
- **Benchmarking** — Built-in utilities for training speed, prediction speed, and model size
- **Rust & Mojo Bindings** — Performance-critical code in Rust via PyO3 and Mojo kernels

## Quick Start

### Tree Models

```python
from shinrin import MondrianTreeRegressor, MondrianForestClassifier
from shinrin import TreeExplainer, explanation

# Train a model
X, y = ...
tree = MondrianTreeRegressor(max_depth=8, random_state=0)
tree.fit(X, y)
predictions = tree.predict(X)

# Get SHAP explanations
explainer = TreeExplainer(tree)
shap_values = explainer.shap_values(X)
# Or use the convenience function:
# explanation(tree, X)  # opens matplotlib visualization
```

### TabM Neural Networks

```python
from shinrin import TabMClassifier, TabMRegressor

# Train a TabM model
model = TabMRegressor(
    hidden_layer_sizes=(256,),
    k=32,               # ensemble size
    random_state=0,
)
model.fit(X, y)
predictions = model.predict(X)

# Classification
clf = TabMClassifier(k=32, max_iter=200)
clf.fit(X, y)
```

### Native Backends

Both Mondrian trees and TabM support interchangeable native backends:

- **Rust** (default) — PyO3/maturin extension for tree models
- **Mojo** — Experimental Mojo port for TabM training kernels

Select the backend with environment variables:

```bash
SHINRIN_BACKEND=mojo python your_script.py          # TabM Mojo backend
SHINRIN_TABM_BACKEND=mojo python your_script.py     # TabM-specific backend
```

## Benchmarks

See [scripts/benchmarks/BENCHMARK.md](scripts/benchmarks/BENCHMARK.md) for detailed benchmark results comparing Shinrin against LightGBM and scikit-learn SGD.

See [scripts/benchmarks/TABM_BENCHMARK.md](scripts/benchmarks/TABM_BENCHMARK.md) for TabM backend comparisons (NumPy vs Mojo vs PyTorch).

To run benchmarks yourself: `python scripts/benchmarks/bench_baselines.py` (or `just bench-backends` for Rust vs Mojo backend comparisons, `just bench-tabm` for TabM backends).

## Installation

```bash
pip install shinrin
```

Optional dependencies:

```bash
pip install shinrin[sklearn]   # scikit-learn for benchmarks and SHAP plotting
pip install shinrin[onnx]      # ONNX export
pip install shinrin[mojo]      # TabM Mojo kernels (`just build-tabm-mojo`)
pip install shinrin[full]      # All optional dependencies
```

### Native backends

The tree/forest internals ship with two interchangeable native backends:

- `rust` (default) – the original pyo3/maturin extension (`shinrin._native`)
- `mojo` – an experimental Mojo port (`shinrin._native_mojo`)

Select the backend with the `SHINRIN_BACKEND` environment variable:

```bash
SHINRIN_BACKEND=mojo python your_script.py
```

The default remains `rust`; the Mojo backend is opt-in while Mojo is in alpha.
Both backends produce identical trees for identical random states (verified by
`tests/test_mojo_parity.py`). Build the Mojo extension with `just build-mojo`
(requires the `mojo` package, e.g. `pip install 'shinrin[mojo]'`).

## API Overview

### Models

#### Tree Models

| Model | Description |
|---|---|
| `MondrianTreeRegressor` | Single Mondrian tree for regression |
| `MondrianTreeClassifier` | Single Mondrian tree for classification |
| `MondrianForestRegressor` | Ensemble of Mondrian trees for regression |
| `MondrianForestClassifier` | Ensemble of Mondrian trees for classification |

#### TabM Neural Networks

| Model | Description |
|---|---|
| `TabMRegressor` | TabM ensemble regressor (drop-in for `MLPRegressor`) |
| `TabMClassifier` | TabM ensemble classifier (drop-in for `MLPClassifier`) |

**TabM parameters:**

| Parameter | Default | Description |
|---|---|---|
| `hidden_layer_sizes` | `(256,)` | Backbone block widths |
| `k` | `32` | Number of ensemble members |
| `solver` | `'adam'` | `'adam'`, `'sgd'`, or `'lbfgs'` |
| `arch_type` | `'tabm'` | `'tabm'`, `'tabm-mini'`, or `'tabm-packed'` |
| `dropout` | `0.1` | Backbone dropout rate |
| `use_embeddings` | `True` | Piecewise-linear + linear embeddings for numeric features |
| `n_bins` | `64` | Quantile bins per numeric feature |
| `d_embedding` | `8` | Embedding width per numeric feature |
| `categorical_indices` | `None` | Columns to treat as categorical |
| `categorical_cardinality_threshold` | `32` | Max unique values for auto-detecting categoricals |

### Explanations (Tree Models)

```python
from shinrin import TreeExplainer, explanation

explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

# Quick visualization (requires matplotlib)
explanation(model, X)
```

### ONNX Export

```python
from shinrin.onnx import to_onnx, save_onnx

# Export to ONNX protobuf
onnx_model = to_onnx(model, X)

# Save to file
save_onnx(model, "model.onnx", X)
```

### Benchmarking

```python
from shinrin.benchmark import (
    benchmark_training,
    benchmark_prediction,
    benchmark_model_size,
    full_benchmark,
    print_benchmark_report,
)

models = {
    "shinrin_tree": MondrianTreeRegressor(max_depth=8),
    "shinrin_forest": MondrianForestRegressor(n_estimators=10, max_depth=8),
    "shinrin_tabm": TabMRegressor(hidden_layer_sizes=(256,), k=32),
}

results = full_benchmark(models, X_train, y_train, X_test)
print_benchmark_report(results)
```

## Test Coverage

All vendored tests are included and passing — these are ported from scikit-garden and skope-rules to verify compatibility. Run `pytest --cov=src/shinrin tests/` for a full coverage report.

TabM parity tests (`tests/test_tabm_parity.py`) verify that the Mojo kernels produce identical results to the NumPy reference implementation. TabM functional tests (`tests/test_tabm.py`) cover training, prediction, and determinism.

## License

MIT