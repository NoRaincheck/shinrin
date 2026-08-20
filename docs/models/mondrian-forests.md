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
