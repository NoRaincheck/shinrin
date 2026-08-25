<div align="center">

# Shinrin

<img src="shinrin.png" alt="Shinrin" width="180"/>

[![PyPI - Version](https://img.shields.io/pypi/v/shinrin.svg)](https://pypi.org/project/shinrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

Shinrin (森林, "forest" in Japanese) is a scikit-learn-compatible library for decision tree and forest models, with Rust and Mojo bindings for performance and ONNX export support.

## Motivation

Since skope-rules and scikit-garden are no longer actively maintained, this project brings them together under one hardened, production-focused umbrella. The goal is to **harden and speed up production training code while keeping the dependency footprint minimal** — dropping large/expensive dependencies like `shap` and `torch`.

- **Tree ensemble library with extensions** — decision rules (`SkopeRules`), optimal decision trees (CORELS, SPOT), and Mondrian forests, with built-in TreeSHAP explanations that do *not* rely on the `shap` library, accelerated by Rust and Mojo kernels. Better together: the `OrdtClassifier` variant routes skope-rules' mined candidates through CORELS' certified selection, outperforming both methods stand-alone
- **Export to standard inference runtimes** — first-class ONNX export for trees and forests

**Roadmap:** better Mojo extension support — including Metal GPU acceleration — and stabilizing the Mojo backend to parity-level reliability.

## Features

- **Mondrian Trees & Forests** — Full scikit-learn API compatibility
- **CORELS Optimal Rule Lists** — Certifiably optimal rule lists for binary data (`CorelsClassifier`), vendored from pycorels with bundled mini-GMP (no system dependency)
- **SPOT Optimal Sparse Decision Trees** (formerly GOSDT) — SParse OpTimal: globally optimized sparse decision trees with reference-ensemble guesses (`SPOTClassifier`, `ThresholdGuessBinarizer`), vendored from gosdt-guesses; optional parallel search workers (`worker_limit`) with no TBB/GMP system dependencies
- **SPOTSET Rashomon Sets of Sparse Trees** (formerly treeFARMS) — Sparse Optimal Rashomon Trees: enumerate *all* near-optimal trees within a bound of the optimum (`SPOTSETClassifier`), co-engineered with SPOT in the same native extension
- **TreeSHAP Explanations** — `TreeExplainer` for single trees and forests with `explanation()` visualization helper
- **ONNX Export** — Export trained trees and forests to ONNX format for deployment
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

### SPOTSET Rashomon Sets of Sparse Trees (formerly treeFARMS)

```python
from shinrin import SPOTSETClassifier, ThresholdGuessBinarizer

# Binarize continuous features via gradient-boosting threshold guesses
X_bin = ThresholdGuessBinarizer().fit_transform(X, y)

# Enumerate all trees within 5% of the optimal regularized objective
clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.05)
clf.fit(X_bin, y)

print(clf.n_trees_)                    # size of the Rashomon set
tree = clf[1]                          # the second tree of the set
print(tree.leaves(), tree.maximum_depth())
trie = clf.get_decision_paths()        # shared decision paths of the whole set
```

### Native Backends

Mondrian trees and forests support interchangeable native backends:

- **Rust** (default) — PyO3/maturin extension for tree models
- **Mojo** — Experimental Mojo port for the Mondrian kernels

Select the backend with an environment variable:

```bash
SHINRIN_BACKEND=mojo python your_script.py     # Mojo backend
```

## Benchmarks

See [scripts/benchmarks/BENCHMARK.md](scripts/benchmarks/BENCHMARK.md) for detailed benchmark results comparing Shinrin against LightGBM and scikit-learn SGD.

See [scripts/benchmarks/ALL_MODELS_BENCHMARK.md](scripts/benchmarks/ALL_MODELS_BENCHMARK.md) for the full all-algorithms benchmark suite, republished at [docs/features/benchmark-results.md](docs/features/benchmark-results.md) (`just bench-all`).

See [scripts/benchmarks/TABARENA_BENCHMARK.md](scripts/benchmarks/TABARENA_BENCHMARK.md) for the same all-algorithms matrix on a core subset of [TabArena](https://arxiv.org/abs/2506.16791)-v0.1 — 13 curated real-world OpenML datasets spanning regression and binary/multiclass classification (`just bench-tabarena`).

See [scripts/benchmarks/GOSDT_BENCHMARK.md](scripts/benchmarks/GOSDT_BENCHMARK.md) for SPOT (formerly GOSDT) vs scikit-learn CART comparisons (`just bench-gosdt`).

See [scripts/benchmarks/ORDT_BENCHMARK.md](scripts/benchmarks/ORDT_BENCHMARK.md) for ORDT — optimal rule-sets from decision trees, combining skope-rules mining with CORELS' certified-optimal selection (ships as `OrdtClassifier`; `just bench-ordt`). Vendoring both pays off: swapping skope-rules' heuristic vote for CORELS' optimal ordering wins on **every dataset tested** (up to +2.6pp test accuracy) while shrinking models to 2–5-clause rule lists.

To run benchmarks yourself: `python scripts/benchmarks/bench_baselines.py`, or use the `just` recipes (`just bench-backends` for Rust vs Mojo tree backends, `just bench-all`, `just bench-tabarena`).

## Installation

```bash
pip install shinrin
```

Optional dependencies:

```bash
pip install shinrin[sklearn]   # scikit-learn for benchmarks and SHAP plotting
pip install shinrin[pandas]    # pandas for SkopeRules / OrdtClassifier
pip install shinrin[onnx]      # ONNX export
pip install shinrin[mojo]      # Mojo kernels (`just build-mojo`)
pip install shinrin[full]      # All core optional dependencies
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
| `SPOTSETClassifier` | Rashomon sets of near-optimal sparse trees (formerly treeFARMS' TREEFARMS) |
| `ThresholdGuessBinarizer` / `NumericBinarizer` | Feature binarizers for SPOT / SPOTSET |

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

Works for tree and forest models. Example:

```python
import onnxruntime as ort
from shinrin import MondrianForestRegressor
from shinrin.onnx import save_onnx

model = MondrianForestRegressor(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
save_onnx(model, "forest.onnx", X_train)

session = ort.InferenceSession("forest.onnx")
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
}

results = full_benchmark(models, X_train, y_train, X_test)
print_benchmark_report(results)
```

## Test Coverage

All vendored tests are included and passing — these are ported from scikit-garden and skope-rules to verify compatibility. Run `pytest --cov=src/shinrin tests/` for a full coverage report.

## License

MIT
