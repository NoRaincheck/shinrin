# TabM

TabM is an efficient MLP-based tabular model that trains an ensemble of
`k` members jointly through BatchEnsemble-style multiplicative adapters,
matching the accuracy of ensembles at a fraction of the training cost.
Shinrin vendors a dependency-free implementation: a pure NumPy reference
trainer plus optional Mojo kernels that run the entire training step
(shuffling, minibatching, Adam or L-BFGS, dropout) natively.

The implementation follows [yandex-research/tabm](https://github.com/yandex-research/tabm)
(Apache-2.0); see NOTICE for attribution details.

## TabMRegressor

Regression with the shared backbone / per-member adapter architecture.

```python
from shinrin import TabMRegressor

model = TabMRegressor(
    hidden_layer_sizes=(256,),
    k=32,
    random_state=0,
)
model.fit(X, y)
```

## TabMClassifier

Binary and multiclass classification with the same architecture.

```python
from shinrin import TabMClassifier

model = TabMClassifier(k=32, max_iter=200)
model.fit(X, y)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_layer_sizes` | `tuple` | `(256,)` | Widths of the shared backbone blocks |
| `solver` | `str` | `"adam"` | `"adam"`, `"sgd"` or `"lbfgs"` |
| `alpha` | `float` | `1e-4` | L2 regularization strength |
| `batch_size` | `int \| "auto"` | `"auto"` | Minibatch size for Adam/SGD |
| `learning_rate_init` | `float` | `1e-3` | Initial learning rate |
| `max_iter` | `int` | `200` | Maximum epochs (or L-BFGS iterations) |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `verbose` | `bool` | `False` | Print per-epoch progress |
| `early_stopping` | `bool` | `False` | Hold out validation data and stop early |
| `validation_fraction` | `float` | `0.1` | Validation split fraction (when `early_stopping`) |
| `n_iter_no_change` | `int` | `10` | Patience in epochs for early stopping |
| `random_state` | `int` | `None` | Random seed for reproducibility |
| `activation` | `str` | `"relu"` | Backbone activation (`"relu"` only) |
| `k` | `int` | `32` | Number of ensemble members |
| `arch_type` | `str` | `"tabm"` | `"tabm"`, `"tabm-mini"` or `"tabm-packed"` |
| `dropout` | `float` | `0.1` | Backbone dropout rate |
| `use_embeddings` | `bool` | `True` | Piecewise-linear + linear embeddings for numeric features |
| `n_bins` | `int` | `64` | Quantile bins per numeric feature for the PLE encoding |
| `d_embedding` | `int` | `8` | Embedding width per numeric feature |
| `use_quantile` | `bool` | `True` | Quantile transform on numeric features |
| `use_asinh` | `bool` | `True` | Inverse hyperbolic sine transform |
| `use_scaler` | `bool` | `True` | Standard scaling |
| `categorical_indices` | `list[int]` | `None` | Force columns (by integer index) to be treated as categorical |
| `categorical_cardinality_threshold` | `int` | `32` | Max unique values for automatic categorical detection; `0` disables auto-detection |
| `quantization` | `str` | `"none"` | `"ternary"` enables BitLinear-style training-aware ternary weight quantization of the shared backbone blocks (see below) |
| `quantization_granularity` | `str` | `"per_row"` | Absmean scale per output row (`"per_row"`) or per matrix (`"per_tensor"`) |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `loss_curve_` | `list[float]` | Loss per epoch/iteration |
| `n_iter_` | `int` | Iterations run |
| `validation_scores_` | `list[float]` | Validation loss per epoch (when `early_stopping`) |
| `preprocessor_` | `_Preprocessor` | Fitted preprocessing state; exposes `categorical_indices_`, `cardinalities_`, `bins_` and value maps |
| `config_` | `TabMConfig` | Resolved architecture configuration |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the model |
| `predict(X)` | Member-averaged predictions |
| `predict_proba(X)` | Class probabilities (classifier only) |
| `score(X, y)` | R² (regression) or accuracy (classification) |
| `partial_fit(X, y)` | One additional Adam/SGD epoch |

## Usage notes

### Categorical data

TabM expects numeric input (`X` must be castable to `float64`). Encode string
or pandas `category` columns as integer codes yourself before fitting, e.g.:

```python
import numpy as np

codes = df["color"].astype("category").cat.codes.to_numpy()
X = np.column_stack([df[["age", "income"]], codes])
```

Columns are treated as categorical when they are explicitly listed in
`categorical_indices` **or** automatically detected by having at most
`categorical_cardinality_threshold` (default 32) unique values. Detected
columns are available after fitting via `model.preprocessor_.categorical_indices_`.

Categorical columns are encoded internally as plain one-hot blocks that feed
straight into the backbone; they bypass all numeric preprocessing and the PLE
embedding layer, so no extra configuration is needed:

```python
model = TabMClassifier(
    k=32,
    categorical_indices=[2, 5],
    max_iter=200,
    random_state=0,
)
```

Unseen values at predict time map to the first fitted category rather than
raising an error.

### Numeric preprocessing and PLE embeddings

For the remaining numeric columns the fit-time pipeline is, in order:
quantile transform → asinh → standard scaling → piecewise-linear encoding
(PLE). With `use_embeddings=True` (the default), each numeric feature's PLE
components — computed over `n_bins` quantile bins — plus its raw value are
projected through learned per-feature embeddings of width `d_embedding`
before entering the backbone.

- `n_bins` and `d_embedding` only take effect when `use_embeddings=True`.
- Set `use_embeddings=False` to skip the PLE encoding and feed scaled numerics
  directly into the backbone (closer to the original TabM recipe).
- Disable individual transforms with `use_quantile=False`, `use_asinh=False`
  or `use_scaler=False` if your data is already well-behaved.

```python
model = TabMRegressor(
    hidden_layer_sizes=(256,),
    k=32,
    use_embeddings=True,
    n_bins=48,
    d_embedding=16,
    random_state=0,
)
```

Fitted bin edges are exposed via `model.preprocessor_.bins_`.

### Ternary quantization (BitLinear)

`quantization="ternary"` quantizes the shared backbone block weights
with the absmean ternary approximation (`{-1, 0, +1} * gamma`,
`gamma = mean(|W|)`) during training, in the style of BitNet b1.58.
Embeddings, adapters, biases and the head stay at full precision;
straight-through gradients keep updating the latent float32 weights.
All arch types (`tabm`, `tabm-mini`, `tabm-packed`) and both backends
are supported, with bit-identical scales between NumPy and Mojo.

```python
model = TabMClassifier(
    k=32,
    max_iter=200,
    random_state=0,
    alpha=0.0,                          # recommended when quantizing
    quantization="ternary",
    quantization_granularity="per_row",  # or "per_tensor"
)
```

!!! warning "Set `alpha=0` for quantized training"
    L2 decay (`alpha > 0`) shrinks the latent weights into the ternary
    dead zone where the effective weight is zero and no gradient flows,
    which often collapses training. The estimator emits a `UserWarning`
    at fit time when this combination is detected.

## Backends

Two trainer backends are available, selected via the
`SHINRIN_TABM_BACKEND` environment variable (`auto`, `numpy` or `mojo`;
`auto` uses Mojo when a prebuilt kernel library is present):

- **numpy** — pure NumPy reference implementation, always available.
- **mojo** — Mojo kernels (`just build-tabm-mojo`) that execute the full
  training step natively; no BLAS requirement.

See [TABM_BENCHMARK.md](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/TABM_BENCHMARK.md)
for measured trade-offs between the backends.
