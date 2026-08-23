# MLP

Shinrin ships scikit-learn compatible plain MLP estimators —
`MLPClassifier` and `MLPRegressor` drop-in replacements for
`sklearn.neural_network.MLP*`. They mirror the scikit-learn parameter
surface, training semantics (losses, learning-rate schedules, momentum,
early stopping) and fitted attributes, while adding:

- a fast NumPy backend plus optional Mojo kernels that run the whole
  training step natively
- an optional piecewise-linear (PLE) embedding for numerical features,
  following the TabM embedding recipe
- dropout and automatic categorical-feature detection

## MLPRegressor / MLPClassifier

```python
from shinrin import MLPRegressor, MLPClassifier

reg = MLPRegressor(hidden_layer_sizes=(100,), random_state=0).fit(X, y)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True).fit(X, y)
```

With the same seed the per-epoch loss curve matches scikit-learn's to
within float32 noise, so existing tuning transfers directly.

### Parameters (scikit-learn compatible)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_layer_sizes` | `tuple` | `(100,)` | Hidden layer widths |
| `activation` | `str` | `"relu"` | `"identity"`, `"logistic"`, `"tanh"` or `"relu"` |
| `solver` | `str` | `"adam"` | `"adam"`, `"sgd"` or `"lbfgs"` |
| `alpha` | `float` | `1e-4` | L2 regularization strength |
| `batch_size` | `int \| "auto"` | `"auto"` | Minibatch size (`"auto"` → 200) |
| `learning_rate` | `str` | `"constant"` | `"constant"`, `"invscaling"` or `"adaptive"` |
| `learning_rate_init` | `float` | `1e-3` | Initial learning rate |
| `power_t` | `float` | `0.5` | Exponent for invscaling |
| `max_iter` | `int` | `200` | Maximum epochs (or L-BFGS iterations) |
| `shuffle` | `bool` | `True` | Shuffle training data each epoch |
| `random_state` | `int` | `None` | Random seed for reproducibility |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `verbose` | `bool` | `False` | Print per-epoch progress |
| `warm_start` | `bool` | `False` | Reuse previous solution when refitting |
| `momentum` | `float` | `0.9` | SGD momentum |
| `nesterovs_momentum` | `bool` | `True` | Nesterov variant for SGD |
| `early_stopping` | `bool` | `False` | Validation-based stopping with best-weight restore |
| `validation_fraction` | `float` | `0.1` | Validation split fraction |
| `beta_1`, `beta_2`, `epsilon` | `float` | Adam moments / epsilon |
| `n_iter_no_change` | `int` | `10` | Patience for stopping / adaptive lr decay |
| `max_fun` | `int` | `15000` | Max iterations for L-BFGS |

### Shinrin extensions

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_embeddings` | `bool` | `False` | Piecewise-linear embeddings for numeric features (see below) |
| `n_bins` | `int` | `64` | Quantile bins per numeric feature for the PLE encoding |
| `d_embedding` | `int` | `8` | Embedding width per numeric feature |
| `dropout` | `float` | `0.0` | Dropout after every hidden layer |
| `use_quantile` | `bool` | `False` | Quantile transform on numeric features |
| `use_asinh` | `bool` | `False` | Inverse hyperbolic sine transform |
| `use_scaler` | `bool` | `False` | Standard scaling |
| `categorical_indices` | `list[int]` | `None` | Force columns (by integer index) categorical |
| `categorical_cardinality_threshold` | `int` | `32` | Max unique values for auto-detection; `0` disables |

Defaults follow scikit-learn exactly: raw features enter the network with
no preprocessing. The PLE recipe below is opt-in.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `classes_` | `ndarray` | Class labels (classifier only) |
| `coefs_`, `intercepts_` | `list[ndarray]` | Per-layer weights (shape `(fan_in, fan_out)` like sklearn) and biases |
| `n_layers_` | `int` | Layer count including the input layer |
| `loss_curve_` | `list[float]` | Loss per epoch/iteration |
| `best_loss_` | `float` | Best training loss seen (not tracked under early stopping, like sklearn) |
| `validation_scores_` | `list[float]` | Validation score per epoch (when `early_stopping`) |
| `best_validation_score_` | `float` | Best validation score (when `early_stopping`) |
| `n_iter_`, `t_` | `int`, `float` | Epochs run / samples seen |
| `out_activation_` | `str` | `"identity"`, `"logistic"` or `"softmax"` |
| `preprocessor_` | `_Preprocessor` | Fitted preprocessing state (`categorical_indices_`, `bins_`, ...) |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the model |
| `predict(X)` | Predicted targets / classes |
| `predict_proba(X)` | Class probabilities (classifier only) |
| `score(X, y)` | R² (regression) or accuracy (classification) |
| `partial_fit(X, y[, classes])` | One additional epoch (no early stopping) |

## PLE embeddings

Setting `use_embeddings=True` routes numerical features through the same
embedding pipeline TabM uses: asinh compression, standardization,
quantile binning (`n_bins`) into piecewise-linear components, then a
trainable per-feature linear projection (`d_embedding`) with ReLU. This
typically buys significant accuracy on tabular data at the cost of a
wider first-layer GEMM:

```python
model = MLPClassifier(
    hidden_layer_sizes=(128,),
    use_embeddings=True,
    use_asinh=True,
    use_scaler=True,
    max_iter=200,
    random_state=0,
).fit(X, y)
```

Categorical columns (detected by cardinality or forced via
`categorical_indices`) bypass the numeric pipeline and feed one-hot
blocks straight into the network. Encode string columns as integer codes
before fitting.

## Backends

Trainer backends are selected via the `SHINRIN_MLP_BACKEND` environment
variable (`auto`, `numpy`, `mojo` or `metal`; `auto` uses the CPU Mojo
kernels when a prebuilt kernel library is present, and Metal is always
opt-in):

- **numpy** — pure NumPy reference implementation, always available;
  supports all solvers.
- **mojo** — CPU Mojo kernels (`just build-mlp-mojo`) executing shuffled
  minibatch Adam epochs (with dropout), L-BFGS and inference natively;
  no BLAS requirement. Non-Adam solvers transparently fall back to
  NumPy. `SHINRIN_MLP_THREADS` overrides the worker count; the kernels
  scale threads down automatically for small minibatches.
- **metal** — experimental Apple-GPU kernels (`just build-mlp-metal`;
  requires the `metal` extra, Xcode 26 with the Metal toolchain and an
  Apple Silicon Mac). Adam only; see the
  [installation notes](../getting-started/installation.md#metal-apple-gpu-backends)
  for requirements and current stability caveats.

See [MLP_BENCHMARK.md](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/MLP_BENCHMARK.md)
for measured comparisons against scikit-learn across both backends.
