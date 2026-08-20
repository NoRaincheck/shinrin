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
