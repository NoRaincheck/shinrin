# Shinrin Documentation

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

## Quick Example

```python
from shinrin import MondrianTreeRegressor, MondrianForestClassifier
from shinrin import TreeExplainer, explanation

# Train a model
tree = MondrianTreeRegressor(max_depth=8, random_state=0)
tree.fit(X, y)
predictions = tree.predict(X)

# Get SHAP explanations
explainer = TreeExplainer(tree)
shap_values = explainer.shap_values(X)
```

## Get Started

- [Installation](getting-started/installation.md) — How to install Shinrin
- [Quick Start](getting-started/quick-start.md) — Get up and running in minutes

## API Reference

See the [API Reference](api-reference.md) for complete documentation of all classes and functions.
