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
  TabM ((128, 128) hidden units, 50 Adam epochs, NumPy reference backend).
- Datasets: synthetic regression (`make_regression`, 4k x 20), binary and
  5-class classification (`make_classification`, 4k x 20); 80/20 split.
- Each cell trains a fresh estimator on float32- and float64-cast data,
  exports via `shinrin.onnx.to_onnx`, and loads the proto into
  onnxruntime (CPU execution provider, intra_op=1 thread).
- Tolerance compares the full test-set outputs: max/mean absolute error,
  classification label agreement, pass/fail against max-abs-error <= 1e-3
  (probabilities and unit-scale predictions) with >= 99.5% label agreement.
- `Struct err` compares onnxruntime against a pure decision-tree traversal
  of the stored tree arrays (the exact semantics the export encodes), i.e.
  exporter fidelity. Mondrian models smooth predictions along the decision
  path as part of the Mondrian-process algorithm, so their native-vs-ORT
  error quantifies that algorithmic gap rather than an export bug.
- TabM graphs are float32-only by design, so f64-trained TabM is served
  through the same f32 graph as f32-trained TabM.
- Speed reports the mean wall-clock per full test-set call after 3 warmup
  calls (timed until >= 0.4 s total or 100 calls). NumPy/BLAS and
  onnxruntime are pinned to one thread on both sides.
- Not every shinrin model has an ONNX exporter: MLP, quantile forests,
  GOSDT, CORELS, SkopeRules, Ordt and TabICL are out of scope here.
- Known numeric floors: Mondrian backends compute at float32 internally
  even for float64 input, capping their achievable agreement near 1e-6;
  sklearn forests average 100 trees in float32 when fed f32 data.

## Regression

Tolerance and speed against `predictions`.

### Tolerance (native vs onnxruntime)

| Dataset | Model | Dtype | Max abs err | Struct err | Mean abs err | Label agree | Check |
|---|---|---|---|---|---|---|---|
| synthetic-reg | MondrianTree | f32 | 4.18e+02 | 0.00e+00 | 1.39e+01 | - | FAIL |
| synthetic-reg | MondrianTree | f64 | 4.18e+02 | 0.00e+00 | 1.39e+01 | - | FAIL |
| synthetic-reg | MondrianForest | f32 | 3.88e+01 | 0.00e+00 | 5.28e+00 | - | FAIL |
| synthetic-reg | MondrianForest | f64 | 3.88e+01 | 0.00e+00 | 5.28e+00 | - | FAIL |
| synthetic-reg | RandomForest | f32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | RandomForest | f64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | ExtraTrees | f32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | ExtraTrees | f64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | - | pass |
| synthetic-reg | TabM | f32 | 1.53e-04 | - | 1.48e-05 | - | pass |
| synthetic-reg | TabM | f64 | 1.53e-04 | - | 1.48e-05 | - | pass |

*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.

### Inference speed

| Dataset | Model | Dtype | Native ms | ORT ms | Speedup |
|---|---|---|---|---|---|
| synthetic-reg | MondrianTree | f32 | 0.513 | 0.0192 | 26.67x |
| synthetic-reg | MondrianTree | f64 | 0.524 | 0.0202 | 25.89x |
| synthetic-reg | MondrianForest | f32 | 10.8 | 0.776 | 13.87x |
| synthetic-reg | MondrianForest | f64 | 10.9 | 0.79 | 13.84x |
| synthetic-reg | RandomForest | f32 | 14.8 | 6.76 | 2.19x |
| synthetic-reg | RandomForest | f64 | 14.6 | 6.86 | 2.12x |
| synthetic-reg | ExtraTrees | f32 | 18 | 8.13 | 2.22x |
| synthetic-reg | ExtraTrees | f64 | 17.7 | 8.06 | 2.20x |
| synthetic-reg | TabM | f32 | 19.9 | 30.7 | 0.65x |
| synthetic-reg | TabM | f64 | 21.5 | 33.1 | 0.65x |

*Speedup*: native_time / ort_time (>1 means onnxruntime is faster).

## Classification

Tolerance and speed against `probabilities + labels`.

### Tolerance (native vs onnxruntime)

| Dataset | Model | Dtype | Max abs err | Struct err | Mean abs err | Label agree | Check |
|---|---|---|---|---|---|---|---|
| synthetic-bin | MondrianTree | f32 | 5.41e-01 | 0.00e+00 | 3.00e-02 | 0.9825 | FAIL |
| synthetic-bin | MondrianTree | f64 | 5.41e-01 | 0.00e+00 | 3.00e-02 | 0.9825 | FAIL |
| synthetic-bin | MondrianForest | f32 | 8.27e-02 | 0.00e+00 | 8.84e-03 | 0.9862 | FAIL |
| synthetic-bin | MondrianForest | f64 | 8.27e-02 | 0.00e+00 | 8.84e-03 | 0.9862 | FAIL |
| synthetic-bin | TabM | f32 | 1.19e-07 | - | 3.00e-08 | 1.0000 | pass |
| synthetic-bin | TabM | f64 | 1.19e-07 | - | 3.00e-08 | 1.0000 | pass |
| synthetic-multi | MondrianTree | f32 | 6.99e-01 | 0.00e+00 | 2.68e-02 | 0.9350 | FAIL |
| synthetic-multi | MondrianTree | f64 | 6.99e-01 | 0.00e+00 | 2.68e-02 | 0.9350 | FAIL |
| synthetic-multi | MondrianForest | f32 | 9.51e-02 | 0.00e+00 | 7.02e-03 | 0.9563 | FAIL |
| synthetic-multi | MondrianForest | f64 | 9.51e-02 | 0.00e+00 | 7.02e-03 | 0.9563 | FAIL |
| synthetic-multi | TabM | f32 | 1.79e-07 | - | 4.73e-09 | 1.0000 | pass |
| synthetic-multi | TabM | f64 | 1.79e-07 | - | 4.73e-09 | 1.0000 | pass |

*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.

### Inference speed

| Dataset | Model | Dtype | Native ms | ORT ms | Speedup |
|---|---|---|---|---|---|
| synthetic-bin | MondrianTree | f32 | 0.573 | 0.045 | 12.71x |
| synthetic-bin | MondrianTree | f64 | 0.661 | 0.0453 | 14.60x |
| synthetic-bin | MondrianForest | f32 | 10.9 | 1.46 | 7.46x |
| synthetic-bin | MondrianForest | f64 | 11.9 | 1.46 | 8.12x |
| synthetic-bin | TabM | f32 | 20.7 | 30.3 | 0.68x |
| synthetic-bin | TabM | f64 | 20.1 | 31.1 | 0.65x |
| synthetic-multi | MondrianTree | f32 | 0.705 | 0.113 | 6.23x |
| synthetic-multi | MondrianTree | f64 | 0.719 | 0.111 | 6.48x |
| synthetic-multi | MondrianForest | f32 | 12.5 | 2.99 | 4.19x |
| synthetic-multi | MondrianForest | f64 | 12.3 | 3.13 | 3.93x |
| synthetic-multi | TabM | f32 | 41.5 | 31.1 | 1.34x |
| synthetic-multi | TabM | f64 | 44.5 | 31.9 | 1.39x |

*Speedup*: native_time / ort_time (>1 means onnxruntime is faster).

## Takeaways

- 10/22 cells meet the tolerance check.
- All 12 failing cells are Mondrian models whose `Struct err` is exactly 0: the ONNX graphs reproduce the
  stored tree semantics bit-for-bit. The gap comes from native Mondrian inference smoothing predictions along the decision path - an algorithmic property, not an export defect.
- f32-trained vs f64-trained native predictions disagree by (max abs): ExtraTrees 2e+01, MondrianForest 6e-06, MondrianTree 6e-05, RandomForest 2e+01, TabM 0e+00. Forests differ most because rounding features/targets before fit changes which splits are chosen (targets here are unnormalized); TabM is exactly 0 because its NumPy backend casts inputs to float64 internally.
- ONNX Runtime is >5% faster in 18 cells and >5% slower in 4 cells (single-threaded CPU); it wins most on deep tree traversal and loses on TabM regression/binary where the NumPy reference batches BLAS-friendly matrix products.

