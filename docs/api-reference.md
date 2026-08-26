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

### SPOT Optimal Sparse Trees (formerly GOSDT)

- `SPOTClassifier` — Globally optimal sparse decision trees with
  reference-ensemble guesses
- `ThresholdGuessBinarizer` — Gradient-boosting threshold binarization
- `NumericBinarizer` — Lossless midpoint binarization
- `Tree` — Parsed optimal tree (via `shinrin._spot`)
- `Status` — Result status enum (via `shinrin._spot`)

See [SPOT Optimal Trees](models/spot.md)

### SPOTSET Rashomon Sets (formerly treeFARMS)

- `SPOTSETClassifier` — Enumerates the Rashomon set of near-optimal sparse
  decision trees; access individual trees via `clf[i]`, the whole set via
  `clf.model_set_`
- `ModelSetContainer` — Lazy container over the extracted set (via
  `shinrin._spotset`)
- `TreeClassifier` — One decoded tree of the set with `predict`/`score`/
  `leaves`/`maximum_depth` helpers

See [SPOTSET Rashomon Sets](models/spotset.md)

## Explanations

- `TreeExplainer` — SHAP explainer for tree models
- `explanation()` — Convenience function for SHAP visualization

### Minimal-Flip Feature Tweaking

- `RashomonFlipSearch(estimator)` — Minimal feature tweaks that flip
  predictions for SPOT, SPOTSET and scikit-learn tree/forest/committee/booster
  classifiers; scopes: `"reference"` (single optimal tree), `"rashomon"`
  (every member of the set), `"ensemble"` (the estimator's own aggregated
  prediction)
  - `.search(X, target=None, scope="rashomon", max_nodes=100_000, time_limit=None)`
    — per-sample minimal-flip search returning `FlipResult` records
- `FlipResult` — Per-sample outcome (`x_new`, `changed_features`,
  `l1_distance`, `success` / `optimal` / `verified`, agreement counts, solver effort)
- `summarize_flip_results(results)` — Batch statistics (success/infeasibility
  rates, distances, solver effort)

See [Minimal-Flip Feature Tweaking](features/minimal-flip-tweaking.md)

## Categorical Features

- `TargetEncoder()` — CatBoost-style target encoder with partition recovery APIs `members()` / `threshold_for_partition()` (`shinrin.TargetEncoder`)
- `to_categorical_tree(model, encoder)` — Recover categorical splits as membership sets; returns a `CategoricalTree` (or list per forest estimator) (`shinrin.categorical`)
- `CategoricalTree` — Tree representation with raw-input `apply()`, `to_text()` rendering, and `to_encoded_thresholds()` round-trip (`shinrin.categorical`)

See [Categorical Features & Target Encoding](features/target-encoding.md)

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
- `ablation_benchmark()` — Fit time and held-out quality per model variant (e.g. two configurations of the same estimator)
- `print_ablation_report()` — Print an ablation table with deltas against the baseline variant
