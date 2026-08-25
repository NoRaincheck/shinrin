# Quick Start

## Basic Usage

### Mondrian Trees

Train a Mondrian Tree Regressor:

```python
from shinrin import MondrianTreeRegressor
import numpy as np

# Generate sample data
X = np.random.rand(100, 4)
y = np.random.rand(100)

# Train the model
tree = MondrianTreeRegressor(max_depth=8, random_state=0)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
```

### Mondrian Forests

Train a Mondrian Forest Classifier:

```python
from shinrin import MondrianForestClassifier

# Generate classification data
X = np.random.rand(200, 4)
y = np.random.randint(0, 3, 200)

# Train the model
forest = MondrianForestClassifier(n_estimators=10, max_depth=8, random_state=0)
forest.fit(X, y)

# Predict classes
predictions = forest.predict(X)
probabilities = forest.predict_proba(X)
```

## SHAP Explanations

Get SHAP values for model interpretability:

```python
from shinrin import TreeExplainer, explanation

# Create an explainer
explainer = TreeExplainer(model)

# Get SHAP values
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

# Quick visualization (requires matplotlib)
explanation(model, X)
```

## ONNX Export

Export models to ONNX format:

```python
from shinrin.onnx import to_onnx, save_onnx

# Export to ONNX protobuf
onnx_model = to_onnx(model, X)

# Save to file
save_onnx(model, "model.onnx", X)
```

## Benchmarking

Compare models with built-in benchmarking utilities:

```python
from shinrin.benchmark import full_benchmark, print_benchmark_report
from shinrin import MondrianTreeRegressor, MondrianForestRegressor

models = {
    "shinrin_tree": MondrianTreeRegressor(max_depth=8),
    "shinrin_forest": MondrianForestRegressor(n_estimators=10, max_depth=8),
}

results = full_benchmark(models, X_train, y_train, X_test)
print_benchmark_report(results)
```

## Optimal Rule Lists & Sparse Trees

For interpretable, certifiably optimal models on binary features:

```python
from shinrin import CorelsClassifier

clf = CorelsClassifier(verbosity=["rulelist"])
clf.fit(X_binary, y, features=["Age<=25", "Prior-Crimes>3"])
print(clf.rl())     # provably optimal if/then rule list
```

For globally optimal sparse trees over continuous features:

```python
from shinrin import SPOTClassifier, ThresholdGuessBinarizer

X_bin = ThresholdGuessBinarizer(n_estimators=20, max_depth=2).fit_transform(X, y)
clf = SPOTClassifier(regularization=0.05, depth_budget=4)
clf.fit(X_bin > 0.5, y)
result = clf.get_result()   # lower_bound == upper_bound certifies optimality
```

See [CORELS Rule Lists](../models/corels.md) and
[SPOT Optimal Trees](../models/spot.md) for full parameter references.
