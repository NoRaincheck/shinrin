# Shinrin

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models, with Rust bindings for performance and ONNX export support.

Since skope-rules and scikit-garden are no longer actively maintained, this project aims to bring them together with extensions for tree models — including SHAP explanations, ONNX export, and benchmarking utilities.

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
- **TabM** — Efficient ensemble MLP trainer with NumPy and native Mojo backends
- **TabICL** — Tabular in-context learning foundation model (torch/NumPy backends)
- **TreeSHAP Explanations** — `TreeExplainer` for single trees and forests with `explanation()` visualization helper
- **ONNX Export** — Export trained models to ONNX format for deployment
- **Benchmarking** — Built-in utilities for training speed, prediction speed, and model size
- **Rust Bindings** — Performance-critical code in Rust via PyO3

## Quick Start

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

## Benchmarks

See [scripts/benchmarks/BENCHMARK.md](scripts/benchmarks/BENCHMARK.md) for detailed benchmark results comparing Shinrin against LightGBM and scikit-learn SGD.

To run benchmarks yourself: `python scripts/benchmarks/bench_baselines.py` (or `just bench-backends` for Rust vs Mojo backend comparisons)

## Installation

```bash
pip install shinrin
```

Optional dependencies:

```bash
pip install shinrin[sklearn]   # scikit-learn for benchmarks and SHAP plotting
pip install shinrin[onnx]      # ONNX export
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

| Model | Description |
|---|---|
| `MondrianTreeRegressor` | Single Mondrian tree for regression |
| `MondrianTreeClassifier` | Single Mondrian tree for classification |
| `MondrianForestRegressor` | Ensemble of Mondrian trees for regression |
| `MondrianForestClassifier` | Ensemble of Mondrian trees for classification |
| `TabMClassifier` / `TabMRegressor` | Ensemble MLP trainers (NumPy / Mojo backends) |
| `TabICLClassifier` / `TabICLRegressor` | Tabular in-context learning estimators (TabICLv2) |

### Explanations

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
}

results = full_benchmark(models, X_train, y_train, X_test)
print_benchmark_report(results)
```

## Test Coverage

All vendored tests are included and passing — these are ported from scikit-garden and skope-rules to verify compatibility. Run `pytest --cov=src/shinrin tests/` for a full coverage report.

## License

MIT