# ONNX Inference Benchmark

Native vs ONNX-runtime inference for every shinrin model that supports
``to_onnx``: each model is trained twice (float32 / float64 data),
exported to ONNX, executed with onnxruntime (CPU), and compared against
the native estimator for numeric agreement and wall-clock speed.

Regenerate locally with:

```bash
uv run python scripts/benchmarks/bench_onnx.py
```

## Environment

| | |
|---|---|
| Date (UTC) | 2026-08-24 |
| OS | Darwin 25.6.0 |
| CPU | Apple M1 Max |
| Cores | 10 |
| Python | 3.14.3 |
| shinrin | 0.2.0 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.0 |
| onnx | 1.22.0 |
| onnxruntime | 1.29.0 |
| Mode | full |

## Methodology

- Models: MondrianTree (depth 16), MondrianForest (20 trees, depth 16),
  RandomForest / ExtraTrees (100 trees, vendored sklearn engine),
  RF-Quantile (50 trees, median baked into the graph),
  MLP ((128, 64) hidden units, 100 Adam epochs),
  TabM ((128, 128) hidden units, 50 Adam epochs; NumPy reference backend),
  Corels and GOSDT (binary-only, on binarized features).
- Datasets: synthetic regression (`make_regression`, 4k x 20), binary and
  5-class classification (`make_classification`, 4k x 20); 80/20 split.
- Each cell trains a fresh estimator on float32- and float64-cast data,
  exports via `shinrin.onnx.to_onnx`, and loads the proto into
  onnxruntime (CPU execution provider, intra_op=1 thread). All exported
  graphs are float32, so f64-trained models ride the f32 deployment path.
- Tolerance compares the full test-set outputs: max/mean absolute error,
  classification label agreement, pass/fail against max-abs-error <= 1e-3
  (probabilities and unit-scale predictions) with >= 99.5% label agreement.
- The exact Mondrian export reproduces native predict/predict_proba
  (including Mondrian-process path smoothing) to float32 round-off;
  generic sklearn-style ensembles round thresholds/values to float32.
- At this dataset scale the exact MondrianForest graph would exceed
  ONNX's 2 GB protobuf limit (selection matrices grow with nodes^2),
  so forests automatically fall back to a plain tree-ensemble export;
  their tolerance rows measure that documented approximation, while
  MondrianTree stays exact.
- Speed reports the mean wall-clock per full test-set call after 3 warmup
  calls (timed until >= 0.4 s total or 100 calls). NumPy/BLAS and
  onnxruntime are pinned to one thread on both sides.
- SkopeRules / Ordt / TabICL are omitted to keep runtime bounded.

## Regression

Tolerance and speed against `predictions`.

### Tolerance (native vs onnxruntime)

| Dataset | Model | Dtype | Export mode | Max abs err | Mean abs err | Label agree | Check |
|---|---|---|---|---|---|---|---|
| synthetic-reg | MondrianTree | f32 | exact | 6.10e-05 | 3.90e-06 | - | pass |
| synthetic-reg | MondrianTree | f64 | exact | 6.10e-05 | 4.08e-06 | - | pass |
| synthetic-reg | MondrianForest | f32 | tree-ensemble | 3.88e+01 | 5.28e+00 | - | FAIL |
| synthetic-reg | MondrianForest | f64 | tree-ensemble | 3.88e+01 | 5.28e+00 | - | FAIL |
| synthetic-reg | RandomForest | f32 | generic | 1.33e-04 | 1.62e-05 | - | pass |
| synthetic-reg | RandomForest | f64 | generic | 1.55e-04 | 1.70e-05 | - | pass |
| synthetic-reg | ExtraTrees | f32 | generic | 2.50e-04 | 1.75e-05 | - | pass |
| synthetic-reg | ExtraTrees | f64 | generic | 1.90e-04 | 1.71e-05 | - | pass |
| synthetic-reg | RF-Quantile | f32 | generic | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | RF-Quantile | f64 | generic | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | MLP | f32 | generic | 2.44e-04 | 1.92e-05 | - | pass |
| synthetic-reg | MLP | f64 | generic | 2.44e-04 | 1.92e-05 | - | pass |
| synthetic-reg | TabM | f32 | generic | 1.53e-04 | 1.48e-05 | - | pass |
| synthetic-reg | TabM | f64 | generic | 1.53e-04 | 1.48e-05 | - | pass |

*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.

*Export mode*: Mondrian graphs are either `exact` (reproduces native predict including path smoothing) or `tree-ensemble` (hard tree structure averaged across trees; the size-guarded fallback used when the exact graph would exceed the protobuf limit). Everything else reports `generic`.

### Inference speed

| Dataset | Model | Dtype | Native ms | ORT ms | Speedup |
|---|---|---|---|---|---|
| synthetic-reg | MondrianTree | f32 | 0.494 | 168 | 0.00x |
| synthetic-reg | MondrianTree | f64 | 0.488 | 168 | 0.00x |
| synthetic-reg | MondrianForest | f32 | 9.68 | 0.776 | 12.48x |
| synthetic-reg | MondrianForest | f64 | 9.97 | 0.802 | 12.43x |
| synthetic-reg | RandomForest | f32 | 13.5 | 6.94 | 1.95x |
| synthetic-reg | RandomForest | f64 | 13.3 | 6.95 | 1.92x |
| synthetic-reg | ExtraTrees | f32 | 15.8 | 8.73 | 1.82x |
| synthetic-reg | ExtraTrees | f64 | 16.1 | 8.85 | 1.82x |
| synthetic-reg | RF-Quantile | f32 | 170 | 5.81e+03 | 0.03x |
| synthetic-reg | RF-Quantile | f64 | 172 | 5.64e+03 | 0.03x |
| synthetic-reg | MLP | f32 | 0.296 | 0.221 | 1.34x |
| synthetic-reg | MLP | f64 | 0.302 | 0.222 | 1.36x |
| synthetic-reg | TabM | f32 | 19 | 29.8 | 0.64x |
| synthetic-reg | TabM | f64 | 18.2 | 29.4 | 0.62x |

*Speedup*: native_time / ort_time (>1 means onnxruntime is faster).

## Classification

Tolerance and speed against `probabilities + labels`.

### Tolerance (native vs onnxruntime)

| Dataset | Model | Dtype | Export mode | Max abs err | Mean abs err | Label agree | Check |
|---|---|---|---|---|---|---|---|
| synthetic-bin | MondrianTree | f32 | exact | 2.38e-07 | 1.35e-08 | 1.0000 | pass |
| synthetic-bin | MondrianTree | f64 | exact | 2.38e-07 | 1.35e-08 | 1.0000 | pass |
| synthetic-bin | MondrianForest | f32 | tree-ensemble | 8.27e-02 | 8.84e-03 | 0.9862 | FAIL |
| synthetic-bin | MondrianForest | f64 | tree-ensemble | 8.27e-02 | 8.84e-03 | 0.9862 | FAIL |
| synthetic-bin | MLP | f32 | generic | 1.19e-07 | 2.58e-08 | 1.0000 | pass |
| synthetic-bin | MLP | f64 | generic | 1.19e-07 | 2.58e-08 | 1.0000 | pass |
| synthetic-bin | TabM | f32 | generic | 1.19e-07 | 3.00e-08 | 1.0000 | pass |
| synthetic-bin | TabM | f64 | generic | 1.19e-07 | 3.00e-08 | 1.0000 | pass |
| synthetic-bin | Corels | f32 | generic | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-bin | Corels | f64 | generic | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-bin | GOSDT | f32 | generic | 0.00e+00 | 0.00e+00 | 1.0000 | pass |
| synthetic-bin | GOSDT | f64 | generic | 0.00e+00 | 0.00e+00 | 1.0000 | pass |
| synthetic-multi | MondrianTree | f32 | exact | 1.19e-07 | 7.82e-09 | 1.0000 | pass |
| synthetic-multi | MondrianTree | f64 | exact | 1.19e-07 | 7.82e-09 | 1.0000 | pass |
| synthetic-multi | MondrianForest | f32 | tree-ensemble | 9.51e-02 | 7.02e-03 | 0.9563 | FAIL |
| synthetic-multi | MondrianForest | f64 | tree-ensemble | 9.51e-02 | 7.02e-03 | 0.9563 | FAIL |
| synthetic-multi | MLP | f32 | generic | 1.19e-07 | 2.09e-09 | 1.0000 | pass |
| synthetic-multi | MLP | f64 | generic | 1.19e-07 | 2.09e-09 | 1.0000 | pass |
| synthetic-multi | TabM | f32 | generic | 1.79e-07 | 4.73e-09 | 1.0000 | pass |
| synthetic-multi | TabM | f64 | generic | 1.79e-07 | 4.73e-09 | 1.0000 | pass |
| synthetic-multi | Corels | f32 | - | - | - | - | - |
| synthetic-multi | Corels | f64 | - | - | - | - | - |
| synthetic-multi | GOSDT | f32 | - | - | - | - | - |
| synthetic-multi | GOSDT | f64 | - | - | - | - | - |

*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.

*Export mode*: Mondrian graphs are either `exact` (reproduces native predict including path smoothing) or `tree-ensemble` (hard tree structure averaged across trees; the size-guarded fallback used when the exact graph would exceed the protobuf limit). Everything else reports `generic`.

### Inference speed

| Dataset | Model | Dtype | Native ms | ORT ms | Speedup |
|---|---|---|---|---|---|
| synthetic-bin | MondrianTree | f32 | 0.55 | 150 | 0.00x |
| synthetic-bin | MondrianTree | f64 | 0.561 | 153 | 0.00x |
| synthetic-bin | MondrianForest | f32 | 9.79 | 0.837 | 11.70x |
| synthetic-bin | MondrianForest | f64 | 9.86 | 0.834 | 11.83x |
| synthetic-bin | MLP | f32 | 0.418 | 0.232 | 1.80x |
| synthetic-bin | MLP | f64 | 0.41 | 0.232 | 1.77x |
| synthetic-bin | TabM | f32 | 19.8 | 28.8 | 0.69x |
| synthetic-bin | TabM | f64 | 19.4 | 28.6 | 0.68x |
| synthetic-bin | Corels | f32 | 0.0807 | 0.0688 | 1.17x |
| synthetic-bin | Corels | f64 | 0.0818 | 0.0643 | 1.27x |
| synthetic-bin | GOSDT | f32 | 0.344 | 0.0656 | 5.24x |
| synthetic-bin | GOSDT | f64 | 0.338 | 0.0645 | 5.24x |
| synthetic-multi | MondrianTree | f32 | 0.681 | 343 | 0.00x |
| synthetic-multi | MondrianTree | f64 | 0.673 | 345 | 0.00x |
| synthetic-multi | MondrianForest | f32 | 11.6 | 0.922 | 12.54x |
| synthetic-multi | MondrianForest | f64 | 11.6 | 0.943 | 12.29x |
| synthetic-multi | MLP | f32 | 0.445 | 0.246 | 1.81x |
| synthetic-multi | MLP | f64 | 0.46 | 0.254 | 1.81x |
| synthetic-multi | TabM | f32 | 41.5 | 29.1 | 1.43x |
| synthetic-multi | TabM | f64 | 40 | 30.4 | 1.32x |
| synthetic-multi | Corels | f32 | - | - | -x |
| synthetic-multi | Corels | f64 | - | - | -x |
| synthetic-multi | GOSDT | f32 | - | - | -x |
| synthetic-multi | GOSDT | f64 | - | - | -x |

*Speedup*: native_time / ort_time (>1 means onnxruntime is faster).

## Takeaways

- 28/38 cells meet the tolerance check.
- All 6 failing cells are MondrianForest exports in `tree-ensemble` mode: above a size guard the exact graph (which reproduces native path smoothing but grows with nodes squared) would exceed ONNX's 2 GB proto limit, so the export falls back to the hard tree structure without smoothing. MondrianTree stays exact and passes.
- 4 cells skipped (binary-only models on multi-class data).
- f32-trained vs f64-trained native predictions disagree by (max abs): Corels 0e+00, ExtraTrees 2e+01, GOSDT 0e+00, MLP 0e+00, MondrianForest 6e-06, MondrianTree 6e-05, RF-Quantile 7e+01, RandomForest 2e+01, TabM 0e+00. Forests differ most because rounding features/targets before fit changes which splits are chosen (targets here are unnormalized); TabM is exactly 0 because its NumPy backend casts inputs to float64 internally.
- ONNX Runtime is >5% faster in 22 cells and >5% slower in 12 cells (single-threaded CPU); it wins most on deep tree traversal and loses where the native path batches BLAS-friendly matrix products.

