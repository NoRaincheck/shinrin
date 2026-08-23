# TabICL

TabICL is a tabular in-context learning foundation model (ICML 2026): a
frozen transformer that predicts labels for test rows directly from the
training set in a single forward pass — no gradient training at all.
Predictions are averaged over an ensemble of normalization, feature-order
and class-order views. Shinrin ships its own implementation of the TabICLv2
architecture following [soda-inria/tabicl](https://github.com/soda-inria/tabicl);
weights are downloaded once from the `jingang/TabICL` Hugging Face
repository, converted to `.npz` archives and shared by every backend.

Install the optional dependencies:

```bash
uv sync --extra tabicl   # torch backend + huggingface-hub download
```

Without torch, the NumPy reference backend still runs but cannot convert a
freshly downloaded checkpoint; convert once on any machine with torch and
copy the generated `.npz` beside the checkpoint.

## TabICLClassifier

```python
from shinrin import TabICLClassifier

clf = TabICLClassifier(n_estimators=8, random_state=42)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

Datasets with more classes than the model's native maximum (10) are handled
automatically through mixed-radix label decomposition and hierarchical
prediction (`support_many_classes=True`).

## TabICLRegressor

The regressor decodes 999 predictive quantiles into a distribution; mean,
median, variance and arbitrary quantiles are available:

```python
from shinrin import TabICLRegressor

reg = TabICLRegressor(n_estimators=8, random_state=42)
reg.fit(X_train, y_train)

mean = reg.predict(X_test)                                  # default: mean
med = reg.predict(X_test, output_type="median")
lo, hi = reg.predict(X_test, output_type="quantiles",
                     alphas=[0.1, 0.9]).T
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | `int` | `8` | Ensemble members (normalization x shuffle views) |
| `norm_methods` | `str \| list` | `None` | Subset of `none`, `power`, `quantile`, `quantile_rtdl`, `robust`; default `["none", "power"]` |
| `feat_shuffle_method` | `str` | `"latin"` | Feature permutation strategy per member |
| `class_shuffle_method` | `str` | `"shift"` | Class-label permutation (classifier only) |
| `outlier_threshold` | `float` | `4.0` | Z-score soft-clipping threshold |
| `softmax_temperature` | `float` | `0.9` | Temperature when averaging logits (classifier only) |
| `average_logits` | `bool` | `True` | Average logits instead of probabilities |
| `support_many_classes` | `bool` | `True` | Mixed-radix + hierarchical prediction above 10 classes |
| `batch_size` | `int` | `8` | Test rows per forward pass (does not change predictions) |
| `kv_cache` | `bool` | `False` | Pre-compute training K/V projections for faster repeated predictions |
| `checkpoint_version` | `str` | v2 file | Checkpoint file name in the HF repo |
| `model_path` | `path` | `None` | Local directory holding the checkpoint |
| `allow_auto_download` | `bool` | `True` | Download the checkpoint when missing |
| `device` | `str` | `None` | Torch device (`"cuda"`); torch backend only |
| `random_state` | `int` | `42` | Seed for ensemble shuffling |
| `backend` | `str` | `"auto"` | `auto`, `torch`, `numpy` or `mojo` |

## Backends

Selected via the `backend` parameter or `SHINRIN_TABICL_BACKEND`
(`auto` prefers torch, then NumPy):

- **numpy** — pure NumPy reference implementation, always available.
- **torch** — own PyTorch implementation with SDPA attention and optional
  GPU inference via `device=`.
- **mojo** — experimental native kernels (`just build-tabicl-mojo`). Run
  the full staged graph natively with SIMD + pthread parallelism and a
  native KV cache (`kv_cache=True`); numeric parity against torch is
  pinned by the opt-in suite (`SHINRIN_TABICL_PARITY_MOJO=1`). Many-class
  hierarchical prediction falls back to torch/numpy at `fit()` time.

See [TABICL_BENCHMARK.md](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/TABICL_BENCHMARK.md)
for benchmark methodology and measured numbers.
