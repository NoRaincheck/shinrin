# GOSDT Optimal Sparse Decision Trees

GOSDT learns *globally optimal* sparse decision trees: instead of growing a
tree greedily, it searches the space of trees with branch-and-bound and
returns the tree minimizing regularized training loss. The vendored
implementation is the canonical one for:

- McTavish et al., *Fast Sparse Decision Tree Optimization via Reference
  Ensembles* (AAAI 2022) — predictions from a blackbox reference model
  (e.g. boosted stumps or a random forest) bound the search,
- Lin et al., *Generalized and Scalable Optimal Sparse Decision Trees*
  (ICML 2020).

Shinrin vendors
[ubc-systopia/gosdt-guesses](https://github.com/ubc-systopia/gosdt-guesses)
(BSD-3-Clause) and compiles its C++ engine into `shinrin._native` with
lock-based replacements for oneTBB and bundled mini-GMP — no TBB or libgmp
system dependencies, so wheels are fully self-contained.

> The search honours `worker_limit`: `1` (default) is single-threaded,
> values above 1 enable parallel branch-and-bound workers, and `0` uses one
> worker per available core. Measured scaling on an M1 Max: ~4x at 8 workers
> on search-heavy workloads (see
> `scripts/benchmarks/GOSDT_BENCHMARK.md`).

## End-to-end pipeline

Continuous features are binarized with threshold guesses from a gradient
boosting ensemble before optimization:

```python
from shinrin import GOSDTClassifier, ThresholdGuessBinarizer

# 1) Binarize via reference-ensemble threshold guesses
enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
X_bin = enc.fit_transform(X, y)

# 2) Fit a certifiably optimal sparse tree
clf = GOSDTClassifier(regularization=0.05, depth_budget=4)
clf.fit(X_bin > 0.5, y)

print(str(clf.trees_[0]))       # the optimal tree
accuracy = clf.score(X_bin > 0.5, y)
```

### Reference ensembles (`y_ref`)

The paper's core idea: hand the optimizer predictions from any blackbox
model to tighten its bounds.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_bin, y)
clf.fit(X_bin > 0.5, y, y_ref=rf.predict(X_bin))
```

`y_ref` must share `y`'s classes; passing it enables the reference
lower-bound machinery automatically.

## GOSDTClassifier

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `regularization` | `float` | `0.05` | Penalty per leaf; bounded below by `1 / n_samples` unless `allow_small_reg` |
| `allow_small_reg` | `bool` | `False` | Permit regularization below `1 / n_samples` (slower search) |
| `depth_budget` | `int \| None` | `None` | Max tree depth (root-only tree = depth 0); `None` is unlimited |
| `time_limit` | `int \| None` | `None` | Seconds; on timeout the best partial model is kept with a warning |
| `balance` | `bool` | `False` | Equalize class importance in the cost matrix |
| `cancellation` | `bool` | `True` | Propagate task cancellations up the dependency graph |
| `look_ahead` | `bool` | `True` | One-step look-ahead bound via scopes |
| `similar_support` | `bool` | `True` | Similar-support bound via the distance index |
| `rule_list` | `bool` | `False` | Constrain solutions to rule lists |
| `non_binary` | `bool` | `False` | Allow non-binary splits (tree parser does not yet render these) |
| `diagnostics` | `bool` | `False` | Print diagnostic traces on internal errors |
| `uncertainty_tolerance` | `float` | `0` | Accepted optimality gap (0 = exact) |
| `upperbound_guess` | `float \| None` | `None` | External upper bound in (0, 1] used for pruning |
| `model_limit` | `int` | `1` | Number of optimal models to extract |
| `worker_limit` | `int` | `1` | `1` = single-threaded; `>1` parallel workers; `0` = one per core |
| `verbose` / `debug` | `bool` | `False` | Progress printing / dump raw inputs for inspection |

### fit / predict

```python
clf.fit(X, y, y_ref=None, input_features=None, cost_matrix=None, feature_map=None)
clf.predict(X, model_number=0)
clf.predict_proba(X, model_number=0)   # one-hot probabilities
clf.score(X, y)                        # inherited accuracy
```

- `X` must be boolean/binary. `y` may be binary or multiclass.
- `cost_matrix`: optional square `n_classes x n_classes` misclassification
  costs.
- `feature_map`: list mapping each original feature to the set of binarized
  columns representing it.

### Result inspection

```python
result = clf.get_result()
# {"models_string": json, "graph_size", "n_iterations",
#  "lower_bound", "upper_bound", "model_loss", "time", "status"}
```

`lower_bound == upper_bound` certifies optimality. `status` is a
[`Status`](#status) enum: `CONVERGED`, `TIMEOUT`, `NON_CONVERGENCE`,
`FALSE_CONVERGENCE`, or `UNINITIALIZED`.

Each extracted model lives in `clf.trees_` as a [`Tree`](#tree) with
`predict` / `predict_proba`.

## ThresholdGuessBinarizer

One-hot encodes numeric features using split thresholds harvested from a
gradient boosting classifier trained on `(X, y)`.

```python
from shinrin import ThresholdGuessBinarizer

enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
X_bin = enc.fit_transform(X, y)
names = enc.get_feature_names_out()   # e.g. "petal width (cm) <= 1.55"
mapping = enc.feature_map()           # original -> binarized column indices
```

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.1` | Gradient boosting learning rate |
| `n_estimators` | `100` | Boosting rounds (use ~20 to keep the search space tractable) |
| `max_depth` | `3` | Boosting tree depth (use ~2 to keep the search space tractable) |
| `random_state` | `0` | Seed |
| `column_elimination` | `True` | Iteratively drop least-important columns while score holds |

With `set_output(transform="pandas")` the transformer returns named
DataFrames like other sklearn transformers.

## NumericBinarizer

Lossless midpoint binarization of numeric features (no reference model):

```python
from shinrin import NumericBinarizer

enc = NumericBinarizer()
X_bin = enc.fit_transform(X)
```

## Tree

A parsed optimal tree. Left branches are the *true* side of the split.

```
{ feature: 0, orig feature: 0,
  [ left child: { prediction: 1, loss: 0.0 },
    right child: { prediction: 0, loss: 0.0 }] }
```

## Status

`IntEnum` mirroring the engine's result status; importable as
`from shinrin._spot import Status`.

## Notes

- Requires the `sklearn` optional extra for the binarizers and metrics.
- Benchmarks vs scikit-learn CART (speed, accuracy, tree sizes, the cost of
  tighter regularization): see [Benchmarking](../features/benchmarking.md#scripts).
