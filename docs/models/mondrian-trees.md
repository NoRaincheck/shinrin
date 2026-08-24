# Mondrian Trees

Mondrian Trees are a type of decision tree built using Mondrian processes, which provide a Bayesian nonparametric approach to tree construction.

## Prediction modes

!!! important "Opinionated default: constant predictions"
    By default (`path_smoothing=False`) predictions are **piecewise-constant
    leaf values**: each sample follows a single root-to-leaf path and
    receives the leaf's stored value. This matches scikit-learn's tree and
    forest behaviour and the plain ONNX `ai.onnx.ml` tree-ensemble export,
    but deliberately deviates from the *pure* Mondrian-process predictor.

    Set `path_smoothing=True` at construction (or pass
    `path_smoothing=True` per `predict()` / `predict_proba()` call) to
    restore the original Mondrian-process weighting, where every node on
    the decision path contributes to the prediction and far-away samples
    shrink towards the root statistics.

## MondrianTreeRegressor

A single Mondrian tree for regression tasks.

```python
from shinrin import MondrianTreeRegressor

tree = MondrianTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    random_state=0,
)
tree.fit(X, y)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_depth` | `int` | `None` | Maximum depth of the tree |
| `min_samples_split` | `int or float` | `2` | Minimum samples required to split a node |
| `random_state` | `int` | `None` | Random seed for reproducibility |
| `path_smoothing` | `bool` | `False` | Use pure Mondrian-process weighted-path predictions instead of constant leaf values |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `n_features_in_` | `int` | Number of features seen during fit |
| `tree_` | `Tree` | The underlying tree structure |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the Mondrian tree to training data |
| `predict(X)` | Predict regression values for X |
| `apply(X)` | Return the index of the leaf for each sample |
| `pred_contribs(X)` | TreeSHAP values with base value appended (see below) |

## SHAP values via `pred_contribs()`

Both Mondrian trees and forests expose `pred_contribs(X)`, which returns
TreeSHAP contributions such that `prediction = base_value + sum(shap_values)`
under the estimator's active prediction mode. The last column/axis holds the
base value: `(n_samples, n_features + 1)` for
regression, `(n_samples, n_features + 1, n_classes)` for classification.

```python
contribs = tree.pred_contribs(X)
shap_values, base_value = contribs[:, :-1], contribs[:, -1]
```

For most purposes the higher-level `TreeExplainer` / `explanation()` helpers
are more convenient; see [TreeSHAP Explanations](../features/treeshap-explanations.md).

## MondrianTreeClassifier

A single Mondrian tree for classification tasks.

```python
from shinrin import MondrianTreeClassifier

tree = MondrianTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    random_state=0,
)
tree.fit(X, y)
```

### Parameters

Same as `MondrianTreeRegressor` with additional:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_weight` | `dict` | `None` | Weights associated with classes |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the Mondrian tree to training data |
| `predict(X)` | Predict class labels for X |
| `predict_proba(X)` | Predict class probabilities for X |
| `apply(X)` | Return the index of the leaf for each sample |
