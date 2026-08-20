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
