# Shinrin

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models, with Rust bindings for performance and ONNX export support.

Since skope-rules and scikit-garden are no longer actively maintained, this project aims to bring them together with extensions for tree models — including SHAP explanations, ONNX export, and benchmarking utilities.

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
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

Benchmarks below compare **Shinrin** (Mondrian trees/forests) against **LightGBM** and **SGD** from scikit-learn.

**Dataset:** 5,000 samples × 20 features (regression) / binary classification

### Regression

| Model | Train Time | Predict Time (1k samples) |
|---|---|---|
| Shinrin Tree (depth=8) | 0.021s | 8.9ms/call |
| Shinrin Forest (n=10) | 0.20s | 89ms/call |
| LightGBM Tree (8 rounds) | 0.05s | 0.17ms/call |
| LightGBM Forest (10 rounds) | 0.06s | — |
| SGDRegressor (100 iters) | 0.003s | 0.04ms/call |
| SGDRegressor (partial_fit, 100 epochs) | 0.06s | — |

### Classification

| Model | Train Time | Predict Time (1k samples) |
|---|---|---|
| Shinrin Tree (depth=8) | 0.021s | 9.6ms/call |
| Shinrin Forest (n=10) | 0.20s | 95ms/call |
| LightGBM Tree (8 rounds) | 0.05s | 0.17ms/call |
| LightGBM Forest (10 rounds) | 0.06s | — |
| SGDClassifier (100 iters) | 0.006s | 0.06ms/call |
| SGDClassifier (partial_fit, 100 epochs) | 0.05s | — |

### Notes

- **Training:** Shinrin tree training is competitive with LightGBM for single trees. Forest training is slower due to Python-level tree construction (Rust optimization pending).
- **Prediction:** LightGBM and SGD are significantly faster at prediction. Shinrin prediction runs in pure Python — Rust-backed prediction is planned.
- **Partial Fit:** SGD supports online/incremental learning via `partial_fit`. Shinrin does not yet support partial fit — this is a planned feature.
- **Shinrin strengths:** TreeSHAP explanations, ONNX export, and Mondrian tree-specific algorithms are unique features not available in LightGBM or SGD.

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

## API Overview

### Models

| Model | Description |
|---|---|
| `MondrianTreeRegressor` | Single Mondrian tree for regression |
| `MondrianTreeClassifier` | Single Mondrian tree for classification |
| `MondrianForestRegressor` | Ensemble of Mondrian trees for regression |
| `MondrianForestClassifier` | Ensemble of Mondrian trees for classification |

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

## License

MIT