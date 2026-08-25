# ONNX Export

Export trained Shinrin models to ONNX format for deployment in any environment that supports ONNX runtime.

## to_onnx()

Convert a trained model to an ONNX model:

```python
from shinrin.onnx import to_onnx

# Export to ONNX protobuf
onnx_model = to_onnx(model, X_example)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | fitted model | The trained Shinrin model to export |
| `X_example` | `ndarray` | Example input data to infer shapes |

### Returns

An ONNX model protobuf.

## save_onnx()

Save a model directly to a file:

```python
from shinrin.onnx import save_onnx

# Save to file
save_onnx(model, "model.onnx", X_example)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | fitted model | The trained Shinrin model to export |
| `path` | `str` | File path to save the ONNX model |
| `X_example` | `ndarray` | Example input data to infer shapes |

## Usage with ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from shinrin.onnx import save_onnx

# Export the model
save_onnx(model, "model.onnx", X)

# Load and run inference with ONNX Runtime
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: X_test})[0]
```

!!! note
    Tree and forest exports use the classic ``ai.onnx.ml``
    ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier`` operators
    (``ai.onnx.ml`` opset 3), supported by every ONNX runtime. All graphs
    accept float32 tensors with a dynamic batch dimension regardless of
    the dtype used for training. Regression graphs expose a
    ``predictions`` vector; classification graphs expose ``probabilities``
    plus ``labels`` (integer class values, or strings when ``class_names``
    is given).

!!! note
    The Mondrian export encoding follows the estimator's
    ``path_smoothing`` prediction mode so exported predictions match native
    ``predict``/``predict_proba`` exactly. Default constant-prediction
    models (``path_smoothing=False``) export as a plain ``ai.onnx.ml``
    tree-ensemble of the hard tree structure — small, fast, and exact.
    Smoothing models (``path_smoothing=True``) export as a self-contained
    standard-domain graph reproducing the Mondrian-process smoothing along
    decision paths to float32 round-off, falling back to the plain
    tree-ensemble for ensembles whose exact graph would exceed ONNX's
    protobuf size limit. Generic sklearn-style forests round thresholds and
    leaf values to float32, keeping agreement near 1e-6.

## Supported Models

| Model | Status |
|---|---|
| `MondrianTreeRegressor` / `MondrianTreeClassifier` | ✅ Exact (`ai.onnx.ml` tree-ensemble; standard-domain graph when `path_smoothing=True`) |
| `MondrianForestRegressor` / `MondrianForestClassifier` | ✅ Exact (`ai.onnx.ml` tree-ensemble; standard-domain graph when `path_smoothing=True`) |
| `RandomForestRegressor` / `ExtraTreesRegressor` | ✅ Supported (`ai.onnx.ml`) |
| `*QuantileRegressor` trees & forests | ✅ Supported (quantile baked in at export) |
| `CorelsClassifier`, `SPOTClassifier`, `OrdtClassifier`, `SkopeRules` | ✅ Supported |

## Importing models with `from_model()`

The reverse direction is supported too: convert a fitted scikit-learn
tree or forest ensemble (or an ONNX model containing `TreeEnsemble`
nodes) into a Mondrian tree or forest that reproduces its predictions.
Mondrian-specific statistics (bounds, tau values, node sample counts)
are rebuilt from `X`/`y`, so the converted model supports incremental
training via `partial_fit`.

```python
from sklearn.ensemble import RandomForestRegressor
from shinrin import MondrianForestRegressor
from shinrin.onnx_import import from_model

rf = RandomForestRegressor().fit(X_train, y_train)

mondrian = from_model(rf, X_train, y_train, MondrianForestRegressor)
mondrian.partial_fit(X_new, y_new)   # continue training online
```

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `model` | fitted sklearn estimator or ONNX `ModelProto` | Source model (needs `tree_` or `estimators_`) |
| `X` | `ndarray` | Training data for the Mondrian statistics rebuild (≥ ~300 samples recommended) |
| `y` | `ndarray` | Training targets |
| `cls` | type | Target Mondrian class, e.g. `MondrianTreeRegressor` |

!!! note
    The conversion preserves the source model's predictions but not its
    internal sampling randomness; subsequent `partial_fit` updates follow
    Mondrian forest semantics.

