# API Reference

## Models

### Mondrian Trees

- `MondrianTreeRegressor` — Single Mondrian tree for regression
- `MondrianTreeClassifier` — Single Mondrian tree for classification

### Mondrian Forests

- `MondrianForestRegressor` — Ensemble of Mondrian trees for regression
- `MondrianForestClassifier` — Ensemble of Mondrian trees for classification

Both forests (and the single Mondrian trees) also expose:

- `pred_contribs(X)` — TreeSHAP values including the base value, so
  `prediction = base_value + sum(shap_values)`
- `pred_anomaly(X)` — Isolation-Forest-style anomaly scores from average
  path length (forests)

See [Mondrian Forests](models/mondrian-forests.md)

### Quantile Regression

- `RandomForestQuantileRegressor` — Random forest with quantile / conditional-std prediction
- `ExtraTreesQuantileRegressor` — Extremely randomized trees variant
- `DecisionTreeQuantileRegressor` — Single-tree quantile regression
- `ExtraTreeQuantileRegressor` — Extremely randomized tree variant
- `RandomForestRegressor` / `ExtraTreesRegressor` — Forest regressors with conditional std support

### Rules

- `SkopeRules` — Rule extraction from tree ensembles
- `Rule` — Extracted rule container
- `replace_feature_name()` — Rename features in a rule

### CORELS Optimal Rule Lists

- `CorelsClassifier` — Certifiably optimal rule lists for binary data
- `OrdtClassifier` — Optimal rule-sets from decision trees (skope-rules
  mining + CORELS selection; variant of `SkopeRules`)
- `RuleList` — Learned rule list (via `shinrin._corels`)
- `load_from_csv` — Load binary CSV datasets (via `shinrin._corels`)

See [CORELS Rule Lists](models/corels.md)

### GOSDT Optimal Sparse Trees

- `GOSDTClassifier` — Globally optimal sparse decision trees with
  reference-ensemble guesses
- `ThresholdGuessBinarizer` — Gradient-boosting threshold binarization
- `NumericBinarizer` — Lossless midpoint binarization
- `Tree` — Parsed optimal tree (via `shinrin._gosdt`)
- `Status` — Result status enum (via `shinrin._gosdt`)

See [GOSDT Optimal Trees](models/gosdt.md)

### Tabular Neural Networks

- `MLPClassifier` — scikit-learn-compatible MLP classifier (NumPy / Mojo backends)
- `MLPRegressor` — scikit-learn-compatible MLP regressor (NumPy / Mojo backends)

See [MLP](models/mlp.md)

- `TabMClassifier` — TabM ensemble classifier (BatchEnsemble-style adapters, NumPy / Mojo backends)
- `TabMRegressor` — TabM ensemble regressor

See [TabM](models/tabm.md)

### TabICL

- `TabICLClassifier` — Tabular in-context learning classifier (TabICLv2)
- `TabICLRegressor` — Tabular in-context learning regressor (quantile decoder)

## Explanations

- `TreeExplainer` — SHAP explainer for tree models
- `explanation()` — Convenience function for SHAP visualization

## ONNX Export

- `to_onnx()` — Convert model to ONNX format (`shinrin.onnx`)
- `save_onnx()` — Save model to ONNX file (`shinrin.onnx`)
- `from_model()` — Import a fitted sklearn tree/forest (or ONNX model) as a Mondrian tree/forest supporting `partial_fit` (`shinrin.onnx_import`)

See [ONNX Export](features/onnx-export.md#importing-models-with-from_model)

## Benchmarking

- `benchmark_training()` — Measure training time
- `benchmark_prediction()` — Measure prediction time
- `benchmark_model_size()` — Measure model size
- `full_benchmark()` — Run all benchmarks
- `print_benchmark_report()` — Print formatted results
- `ablation_benchmark()` — Fit time and held-out quality per model variant (e.g. with vs without ternary quantization)
- `print_ablation_report()` — Print an ablation table with deltas against the baseline variant
