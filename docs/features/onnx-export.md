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

## Supported Models

| Model | Status |
|---|---|
| `MondrianTreeRegressor` | ✅ Supported |
| `MondrianTreeClassifier` | ✅ Supported |
| `MondrianForestRegressor` | ✅ Supported |
| `MondrianForestClassifier` | ✅ Supported |
| `TabMRegressor` | ✅ Supported |
| `TabMClassifier` | ✅ Supported |

## TabM Export

TabM models are exported as a single self-contained graph (opset 15,
standard-domain ops only): preprocessing (quantile/asinh/scaler transforms,
piecewise-linear encoding, categorical one-hot), the PLE embedding layer,
the BatchEnsemble backbone and the ensemble-averaged head are all baked in.
Deployment only needs the `.onnx` file and raw feature vectors — no
scikit-learn or shinrin code at inference time.

```python
import numpy as np
import onnxruntime as ort
from shinrin import TabMRegressor
from shinrin.onnx import save_onnx

model = TabMRegressor(hidden_layer_sizes=(256,), k=32, random_state=0)
model.fit(X_train, y_train)

save_onnx(model, "tabm.onnx", X_train)

session = ort.InferenceSession("tabm.onnx")
predictions = session.run(None, {"X": X_test.astype(np.float32)})[0]
```

Outputs:

| Task | Outputs |
|---|---|
| Regression | `predictions`: `(n_samples,)` or `(n_samples, n_outputs)` float32 |
| Classification | `probabilities`: `(n_samples, n_classes)` float32 and `labels`: int64 indices into `classes_` (or strings matching `class_names`) |

Notes:

- The input tensor `X` is float32 with a dynamic batch dimension; any
  batch size works regardless of the export-time data.
- All three architectures (`tabm`, `tabm-mini`, `tabm-packed`) are
  supported; dropout is inference-inactive and therefore not exported.
- Categorical features are bucketized by comparing against the fitted
  category values: unseen values below the smallest known category map to
  one-hot index 0 (values above the largest map to the last index).
- Numerical precision matches the NumPy reference to float32 round-off
  (verified against onnxruntime in `tests/test_tabm_onnx.py`).
