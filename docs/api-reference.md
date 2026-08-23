# API Reference

## Models

### Mondrian Trees

- `MondrianTreeRegressor` — Single Mondrian tree for regression
- `MondrianTreeClassifier` — Single Mondrian tree for classification

### Mondrian Forests

- `MondrianForestRegressor` — Ensemble of Mondrian trees for regression
- `MondrianForestClassifier` — Ensemble of Mondrian trees for classification

### CORELS Optimal Rule Lists

- `CorelsClassifier` — Certifiably optimal rule lists for binary data
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

### TabICL

- `TabICLClassifier` — Tabular in-context learning classifier (TabICLv2)
- `TabICLRegressor` — Tabular in-context learning regressor (quantile decoder)

## Explanations

- `TreeExplainer` — SHAP explainer for tree models
- `explanation()` — Convenience function for SHAP visualization

## ONNX Export

- `to_onnx()` — Convert model to ONNX format
- `save_onnx()` — Save model to ONNX file

## Benchmarking

- `benchmark_training()` — Measure training time
- `benchmark_prediction()` — Measure prediction time
- `benchmark_model_size()` — Measure model size
- `full_benchmark()` — Run all benchmarks
- `print_benchmark_report()` — Print formatted results
- `ablation_benchmark()` — Fit time and held-out quality per model variant (e.g. with vs without ternary quantization)
- `print_ablation_report()` — Print an ablation table with deltas against the baseline variant
