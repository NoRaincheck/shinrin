# Mondrian Trees

Mondrian Trees are a type of decision tree built using Mondrian processes, which provide a Bayesian nonparametric approach to tree construction.

## MondrianTreeRegressor

A single Mondrian tree for regression tasks.

```python
from shinrin import MondrianTreeRegressor

tree = MondrianTreeRegressor(
    max_depth=8,
    min_weight=0.001,
    random_state=0,
)
tree.fit(X, y)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_depth` | `int` | `8` | Maximum depth of the tree |
| `min_weight` | `float` | `0.001` | Minimum leaf weight |
| `random_state` | `int` | `None` | Random seed for reproducibility |

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
TreeSHAP contributions such that `prediction = base_value + sum(shap_values)`.
The last column/axis holds the base value: `(n_samples, n_features + 1)` for
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
    max_depth=8,
    min_weight=0.001,
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
