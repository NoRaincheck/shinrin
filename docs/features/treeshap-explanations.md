# TreeSHAP Explanations

Shinrin provides TreeSHAP-based explanations for Mondrian Trees and Forests through the `TreeExplainer` class.

## TreeExplainer

The `TreeExplainer` computes SHAP (SHapley Additive exPlanations) values for tree models.

```python
from shinrin import TreeExplainer

# Create an explainer
explainer = TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X)

# Get the expected value (base score)
expected_value = explainer.expected_value
```

## explanation() Helper

For quick visualization, use the convenience `explanation()` function:

```python
from shinrin import explanation

# Opens a matplotlib visualization
explanation(model, X)
```

## SHAP Value Interpretation

SHAP values represent the contribution of each feature to a specific prediction:

- **Positive values** push the prediction higher than the expected value
- **Negative values** push the prediction lower than the expected value
- The sum of SHAP values plus the expected value equals the model output

## Example

```python
import numpy as np
from shinrin import MondrianForestRegressor, TreeExplainer

# Train a model
X = np.random.rand(100, 4)
y = np.random.rand(100)
model = MondrianForestRegressor(n_estimators=10, max_depth=8, random_state=0)
model.fit(X, y)

# Create explainer and compute values
explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Analyze feature importance
import pandas as pd
feature_importance = np.abs(shap_values).mean(axis=0)
print(pd.Series(feature_importance, index=[f"Feature {i}" for i in range(4)]))
```
