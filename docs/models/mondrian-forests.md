# Mondrian Forests

Mondrian Forests are ensembles of Mondrian Trees that provide improved accuracy through aggregation of multiple tree predictions.

## MondrianForestRegressor

An ensemble of Mondrian trees for regression.

```python
from shinrin import MondrianForestRegressor

forest = MondrianForestRegressor(
    n_estimators=10,
    max_depth=8,
    min_weight=0.001,
    random_state=0,
)
forest.fit(X, y)
predictions = forest.predict(X)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | `int` | `10` | Number of trees in the forest |
| `max_depth` | `int` | `8` | Maximum depth of each tree |
| `min_weight` | `float` | `0.001` | Minimum leaf weight |
| `random_state` | `int` | `None` | Random seed for reproducibility |

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
    max_depth=8,
    min_weight=0.001,
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
