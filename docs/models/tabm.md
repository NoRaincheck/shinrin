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
| `early_stopping` | `bool` | `False` | Hold out validation data and stop early |
| `random_state` | `int` | `None` | Random seed for reproducibility |
| `k` | `int` | `32` | Number of ensemble members |
| `arch_type` | `str` | `"tabm"` | `"tabm"`, `"tabm-mini"` or `"tabm-packed"` |
| `dropout` | `float` | `0.1` | Backbone dropout rate |
| `use_embeddings` | `bool` | `True` | Piecewise-linear + linear embeddings for numeric features |
| `n_bins` | `int` | `64` | Quantile bins per numeric feature for the PLE encoding |
| `d_embedding` | `int` | `8` | Embedding width per numeric feature |
| `use_quantile` | `bool` | `True` | Quantile transform on numeric features |
| `use_asinh` | `bool` | `True` | Inverse hyperbolic sine transform |
| `use_scaler` | `bool` | `True` | Standard scaling |
| `categorical_indices` | `list` | `None` | Force columns to be treated as categorical |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `loss_curve_` | `list[float]` | Loss per epoch/iteration |
| `n_iter_` | `int` | Iterations run |
| `validation_scores_` | `list[float]` | Validation loss per epoch (when `early_stopping`) |
| `bins_` | `list[np.ndarray]` | Fitted quantile bin edges |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the model |
| `predict(X)` | Member-averaged predictions |
| `predict_proba(X)` | Class probabilities (classifier only) |
| `score(X, y)` | R² (regression) or accuracy (classification) |
| `partial_fit(X, y)` | One additional Adam/SGD epoch |

## Backends

Two trainer backends are available, selected via the
`SHINRIN_TABM_BACKEND` environment variable (`auto`, `numpy` or `mojo`;
`auto` uses Mojo when a prebuilt kernel library is present):

- **numpy** — pure NumPy reference implementation, always available.
- **mojo** — Mojo kernels (`just build-tabm-mojo`) that execute the full
  training step natively; no BLAS requirement.

See [TABM_BENCHMARK.md](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/TABM_BENCHMARK.md)
for measured trade-offs between the backends.
