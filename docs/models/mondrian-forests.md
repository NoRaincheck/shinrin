# Mondrian Forests

Mondrian Forests are ensembles of Mondrian Trees that provide improved accuracy through aggregation of multiple tree predictions.

## Prediction modes

!!! important "Opinionated default: constant predictions"
    By default (`path_smoothing=False`) each tree predicts a
    **piecewise-constant leaf value** and the forest averages across trees —
    the same behaviour as scikit-learn's forests, and exactly what the plain
    ONNX `ai.onnx.ml` tree-ensemble export computes. Set
    `path_smoothing=True` to use the *pure* Mondrian-process predictor,
    where every node on each tree's decision path contributes with a
    Mondrian-process weight (see
    [Mondrian Trees](mondrian-trees.md#prediction-modes)).

## MondrianForestRegressor

An ensemble of Mondrian trees for regression.

```python
from shinrin import MondrianForestRegressor

forest = MondrianForestRegressor(
    n_estimators=10,
    max_depth=None,
    min_samples_split=2,
    random_state=0,
)
forest.fit(X, y)
predictions = forest.predict(X)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | `int` | `10` | Number of trees in the forest |
| `max_depth` | `int` | `None` | Maximum depth of each tree |
| `min_samples_split` | `int or float` | `2` | Minimum samples required to split a node |
| `bootstrap` | `bool` | `False` | Bootstrap samples when building trees |
| `random_state` | `int` | `None` | Random seed for reproducibility |
| `path_smoothing` | `bool` | `False` | Use pure Mondrian-process weighted-path predictions instead of constant leaf values |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `n_features_in_` | `int` | Number of features seen during fit |
| `estimators_` | `list` | List of trained Mondrian trees |

## MondrianForestClassifier

An ensemble of Mondrian trees for classification.

```python
from shinrin import MondrianForestClassifier

forest = MondrianForestClassifier(
    n_estimators=10,
    max_depth=None,
    min_samples_split=2,
    random_state=0,
)
forest.fit(X, y)
predictions = forest.predict(X)
probabilities = forest.predict_proba(X)
```

### Parameters

Same as `MondrianForestRegressor` with additional:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_weight` | `dict` | `None` | Weights associated with classes |

### Attributes & Methods

Same as `MondrianForestRegressor` with classification-specific methods:

| Method | Description |
|---|---|
| `predict(X)` | Predict class labels for X |
| `predict_proba(X)` | Predict class probabilities for X |
| `apply(X)` | Return the index of the leaf for each sample |

## SHAP values and anomaly scores

Forests (and single trees) expose two extension methods beyond the
scikit-learn API:

### `pred_contribs(X)`

TreeSHAP values averaged across trees, with the base value appended so
that `prediction = base_value + sum(shap_values)`. Shape is
`(n_samples, n_features + 1)` for regression or
`(n_samples, n_features + 1, n_classes)` for classification.

```python
contribs = forest.pred_contribs(X)
shap_values, base_value = contribs[:, :-1], contribs[:, -1]
```

See [TreeSHAP Explanations](../features/treeshap-explanations.md) for the
higher-level explainer API.

### `pred_anomaly(X[, n_train])`

Isolation-Forest-style anomaly scores from the average path length of each
sample through the forest: ~0.5 normal, near 1.0 anomalous (short paths).
Pass `n_train` if the forest was trained in batches via `partial_fit` so
scores are normalized against the full training size.

```python
scores = forest.pred_anomaly(X)
outliers = X[scores > 0.6]
```
