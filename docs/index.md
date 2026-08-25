# Shinrin Documentation

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models and tabular neural networks, with Rust and Mojo bindings for performance and ONNX export support.

Since skope-rules and scikit-garden are no longer actively maintained, this project aims to bring them together with extensions for tree models — including rule extraction (SkopeRules), certifiably optimal rule lists (CORELS), globally optimal sparse decision trees (SPOT, formerly GOSDT) and optimal rule-sets (ORDT) — plus torch-free tabular neural networks (MLP, TabM) and the TabICL in-context foundation model, SHAP explanations, ONNX export, and benchmarking utilities. The vendored CORELS and SPOT (formerly GOSDT) engines compile into the native extension with bundled mini-GMP and no TBB, so `pip install` needs no system libraries. The vendored parts also compound: routing skope-rules' mined candidates through CORELS' certified-optimal selection (ORDT) beats both methods stand-alone across our benchmarks (up to +2.6pp accuracy at 2–5-clause model sizes).

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
- **CORELS Optimal Rule Lists** — Certifiably optimal rule lists (`CorelsClassifier`) with bundled mini-GMP, no system dependency
- **SPOT Optimal Sparse Trees** (formerly GOSDT) — Globally optimized trees with reference-ensemble guesses (`SPOTClassifier`, `ThresholdGuessBinarizer`)
- **SPOTSET Rashomon Sets** (formerly treeFARMS) — All near-optimal trees within a configurable bound of the optimum (`SPOTSETClassifier`)
- **Rule Extraction & ORDT** — `SkopeRules` plus the `OrdtClassifier` variant that routes mined candidates through CORELS' certified selection
- **Tabular Neural Networks** — scikit-learn-compatible `MLPClassifier`/`MLPRegressor` and `TabMClassifier`/`TabMRegressor` with optional PLE embeddings, ternary weight quantization (BitLinear), and Mojo-accelerated training
- **TabICL** — Tabular in-context learning foundation model (torch/NumPy/Mojo backends)
- **TreeSHAP Explanations** — `TreeExplainer` for single trees and forests with `explanation()` visualization helper
- **ONNX Export** — Export trained models to ONNX format for deployment
- **Benchmarking** — Built-in utilities for training speed, prediction speed, and model size
- **Rust & Mojo Bindings** — Performance-critical code in Rust via PyO3 and Mojo kernels

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
