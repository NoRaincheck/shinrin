# API Reference

## Models

### Mondrian Trees

- `MondrianTreeRegressor` — Single Mondrian tree for regression
- `MondrianTreeClassifier` — Single Mondrian tree for classification

### Mondrian Forests

- `MondrianForestRegressor` — Ensemble of Mondrian trees for regression
- `MondrianForestClassifier` — Ensemble of Mondrian trees for classification

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
