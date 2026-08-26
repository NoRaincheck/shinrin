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

## Categorical features & `BRANCH_MEMBER` (opset 5)

When categorical columns are handled by training on target-encoded values
(see [`TargetEncoder`](target-encoding.md)), the default export has a
drawback: encoded-threshold splits only make sense together with the
encoder, so the deployed graph must ship the encoder too.

Passing the fitted encoder via `encoder=` switches to an `ai.onnx.ml`
opset-5 `TreeEnsemble` where every categorical split becomes a
`BRANCH_MEMBER` node testing **raw category-code membership**
(`x_color in {0, 2}` instead of `x_color_enc <= 0.37`). The graph then
consumes your original input feature convention — numeric columns and raw
integer category codes — with no encoder at inference time:

```python
import numpy as np
import shinrin

X_raw = ...  # column 0 holds integer category codes
enc = shinrin.TargetEncoder(categorical_features=[0]).fit(X_raw, y)
model = shinrin.MondrianForestRegressor(n_estimators=20).fit(
    enc.transform(X_raw), y
)

onnx_model = shinrin.to_onnx(model, X=X_raw, encoder=enc)
shinrin.save_onnx(model, "model.onnx", X=X_raw, encoder=enc)
```

Notes:

- Only prefixes of the encoding-sorted categories are representable as a
  single encoded threshold, so recovery is exact for any model trained on
  encoded data.
- Unseen categories: at inference the ONNX graph routes codes that were
  absent during training through the false branches of membership tests.
  Native predict would encode them to the prior; if exact parity on
  unseen categories matters, re-fit with those categories present.
- Mondrian models with `path_smoothing=True`: smoothing cannot be
  represented in a tree ensemble, so `encoder=` exports the hard tree
  structure and emits a `UserWarning`. Pass `approximate=False` to build
  the exact smoothing graph (without member splits) instead.
- Quantile models do not support `encoder=`.

The exported model carries a `shinrin_treeensemble_export="member-v5"`
metadata property.

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

