<div align="center">

# Shinrin

<img src="shinrin.png" alt="Shinrin" width="180"/>

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models and tabular neural networks, with Rust and Mojo bindings for performance and ONNX export support.

## Motivation

Since skope-rules and scikit-garden are no longer actively maintained, this project brings them together under one hardened, production-focused umbrella. The goal is to **harden and speed up production training code while keeping the dependency footprint minimal** — dropping large/expensive dependencies like `shap` and `torch`.

- **Tree ensemble library with extensions** — decision rules (`SkopeRules`), optimal decision trees (CORELS, SPOT), and Mondrian forests, with built-in TreeSHAP explanations that do *not* rely on the `shap` library, accelerated by Rust and Mojo kernels. Better together: the `OrdtClassifier` variant routes skope-rules' mined candidates through CORELS' certified selection, outperforming both methods stand-alone
- **Tabular neural networks without torch** — MLPs and TabM train entirely on NumPy or Mojo kernels; no PyTorch required
- **Export to standard inference runtimes** — first-class ONNX export for trees, forests, and TabM

**Roadmap:** better Mojo extension support — including Metal GPU acceleration — and stabilizing the Mojo backend to parity-level reliability.

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
- **CORELS Optimal Rule Lists** — Certifiably optimal rule lists for binary data (`CorelsClassifier`), vendored from pycorels with bundled mini-GMP (no system dependency)
- **SPOT Optimal Sparse Decision Trees** (formerly GOSDT) — SParse OpTimal: globally optimized sparse decision trees with reference-ensemble guesses (`SPOTClassifier`, `ThresholdGuessBinarizer`), vendored from gosdt-guesses; optional parallel search workers (`worker_limit`) with no TBB/GMP system dependencies
- **Tabular Neural Networks** — scikit-learn compatible `MLPClassifier`/`MLPRegressor` and `TabMClassifier`/`TabMRegressor` with optional PLE embeddings, training-aware ternary weight quantization (BitLinear), and Mojo-accelerated training
- **TabM Neural Networks** — Parameter-efficient ensemble MLPs for tabular data with BatchEnsemble-style multiplicative adapters (ICLR 2025)
- **TabICL** — Tabular in-context learning foundation model (torch/NumPy/Mojo backends)
- **TreeSHAP Explanations** — `TreeExplainer` for single trees and forests with `explanation()` visualization helper
- **ONNX Export** — Export trained trees, forests, and TabM models to ONNX format for deployment
- **Benchmarking** — Built-in utilities for training speed, prediction speed, model size, and before/after feature ablations
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

### CORELS Optimal Rule Lists

```python
from shinrin import CorelsClassifier

# Binary features, binary classification — provably optimal rule list
clf = CorelsClassifier(c=0.01, verbosity=["rulelist"])
clf.fit(X, y, features=["feature1", "feature2"])
print(clf.rl())          # human-readable optimal rule list
predictions = clf.predict(X)
```

### SPOT Optimal Sparse Decision Trees (formerly GOSDT)

```python
from shinrin import SPOTClassifier, ThresholdGuessBinarizer

# Binarize continuous features via gradient-boosting threshold guesses
X_bin = ThresholdGuessBinarizer().fit_transform(X, y)

# Optionally guide the search with a blackbox reference ensemble
clf = SPOTClassifier(regularization=0.05, depth_budget=4)
clf.fit(X_bin, y)                      # or: clf.fit(X_bin, y, y_ref=y_ref)
print(str(clf.trees_[0]))              # globally optimal tree
accuracy = clf.score(X_bin, y)
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

See [scripts/benchmarks/MLP_BENCHMARK.md](scripts/benchmarks/MLP_BENCHMARK.md) for MLP comparisons against scikit-learn across the NumPy and Mojo backends (`just bench-mlp`).

See [scripts/benchmarks/BITLINEAR_BENCHMARK.md](scripts/benchmarks/BITLINEAR_BENCHMARK.md) for ternary quantization (BitLinear) ablations — training-aware QAT for MLP/TabM on both backends plus TabICL post-training quantization (`just bench-bitlinear`).

See [scripts/benchmarks/ALL_MODELS_BENCHMARK.md](scripts/benchmarks/ALL_MODELS_BENCHMARK.md) for the full all-algorithms benchmark suite, republished at [docs/features/benchmark-results.md](docs/features/benchmark-results.md) (`just bench-all`).

See [scripts/benchmarks/GOSDT_BENCHMARK.md](scripts/benchmarks/GOSDT_BENCHMARK.md) for SPOT (formerly GOSDT) vs scikit-learn CART comparisons (`just bench-gosdt`).

See [scripts/benchmarks/ORDT_BENCHMARK.md](scripts/benchmarks/ORDT_BENCHMARK.md) for ORDT — optimal rule-sets from decision trees, combining skope-rules mining with CORELS' certified-optimal selection (ships as `OrdtClassifier`; `just bench-ordt`). Vendoring both pays off: swapping skope-rules' heuristic vote for CORELS' optimal ordering wins on **every dataset tested** (up to +2.6pp test accuracy) while shrinking models to 2–5-clause rule lists.

See [scripts/benchmarks/TABICL_BENCHMARK.md](scripts/benchmarks/TABICL_BENCHMARK.md) for TabICL inference benchmarks across the NumPy/torch/Mojo backends, including predict throughput, ternary PTQ ablation and batch-size/KV-cache sweeps (`python scripts/benchmarks/bench_tabicl.py --backend mojo --quant-ablation --cache-sweep`).

To run benchmarks yourself: `python scripts/benchmarks/bench_baselines.py`, or use the `just` recipes (`just bench-backends` for Rust vs Mojo tree backends, `just bench-mlp`, `just bench-bitlinear`, `just bench-all`). TabM and TabICL backend comparisons have no recipe: `python scripts/benchmarks/bench_tabm.py` and `python scripts/benchmarks/bench_tabicl.py`.

## Installation

```bash
pip install shinrin
```

Optional dependencies:

```bash
pip install shinrin[sklearn]   # scikit-learn for benchmarks and SHAP plotting
pip install shinrin[pandas]    # pandas for SkopeRules / OrdtClassifier
pip install shinrin[onnx]      # ONNX export
pip install shinrin[tabicl]    # TabICL torch backend + checkpoint download
pip install shinrin[mojo]      # Mojo kernels (`just build-*-mojo`)
pip install shinrin[full]      # All core optional dependencies
```

Benchmark-only extras (`tabm-bench`, `tabicl-bench`) pull in PyTorch and the
upstream reference packages for `bench_tabm.py --with-torch` /
`bench_tabicl.py --with-upstream` comparisons.

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

#### Trees & Forests

| Model | Description |
|---|---|
| `MondrianTreeRegressor` / `MondrianTreeClassifier` | Single Mondrian tree |
| `MondrianForestRegressor` / `MondrianForestClassifier` | Ensemble of Mondrian trees |
| `RandomForestQuantileRegressor` / `ExtraTreesQuantileRegressor` | Quantile regression forests |
| `DecisionTreeQuantileRegressor` / `ExtraTreeQuantileRegressor` | Single-tree quantile regression |

#### Rules & Optimal Trees

| Model | Description |
|---|---|
| `SkopeRules` | Rule extraction from tree ensembles |
| `OrdtClassifier` | Optimal rule-sets: skope-rules mining + CORELS certified selection |
| `CorelsClassifier` | Certifiably optimal rule lists for binary data |
| `SPOTClassifier` | Globally optimal sparse decision trees (formerly GOSDTClassifier) |
| `ThresholdGuessBinarizer` / `NumericBinarizer` | Feature binarizers for SPOT |

#### Tabular Neural Networks

| Model | Description |
|---|---|
| `MLPClassifier` / `MLPRegressor` | scikit-learn-compatible drop-in MLPs (NumPy / Mojo backends) |
| `TabMClassifier` / `TabMRegressor` | Ensemble MLP trainers, BatchEnsemble-style adapters (NumPy / Mojo backends) |
| `TabICLClassifier` / `TabICLRegressor` | Tabular in-context learning estimators (TabICLv2) |

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
| `quantization` | `'none'` | `'ternary'` enables BitLinear ternary weight quantization (set `alpha=0`) |
| `quantization_granularity` | `'per_row'` | Absmean scale per output row or per matrix (`'per_tensor'`) |

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

Works for tree/forest models and TabM. TabM exports a self-contained
graph (preprocessing, embeddings, ensemble backbone, averaged head) that
runs on raw feature vectors with any batch size:

```python
import onnxruntime as ort
from shinrin import TabMRegressor
from shinrin.onnx import save_onnx

model = TabMRegressor(hidden_layer_sizes=(256,), k=32, random_state=0)
model.fit(X_train, y_train)
save_onnx(model, "tabm.onnx", X_train)

session = ort.InferenceSession("tabm.onnx")
predictions = session.run(None, {"X": X_test.astype(np.float32)})[0]
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

TabM parity tests (`tests/test_tabm_parity.py`) verify that the Mojo kernels produce identical results to the NumPy reference implementation. TabM functional tests (`tests/test_tabm.py`) cover training, prediction, and determinism. TabM ONNX export tests (`tests/test_tabm_onnx.py`) verify onnxruntime inference parity for all architectures, tasks, and preprocessing configurations.

## License

MIT
