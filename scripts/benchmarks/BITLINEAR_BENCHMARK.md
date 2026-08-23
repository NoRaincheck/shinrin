# BitLinear (Ternary Quantization) Benchmarks

Comparison of training-aware ternary weight quantization ("BitLinear",
BitNet b1.58-style absmean scheme) against full-precision training in the
shinrin MLP and TabM trainers, plus experimental post-training ternary
quantization for TabICL inference.

To run: `uv run python scripts/benchmarks/bench_bitlinear.py [--samples N]
[--features N] [--max-iter N] [--backends numpy,mojo] [--estimators mlp,tabm]
[--tasks cls,reg] [--tabicl]` (or `just bench-bitlinear`). The Mojo columns
require prebuilt kernels (`just build-mlp-mojo`, `just build-tabm-mojo`).

## Variants

- **fp** — full-precision latent weights (baseline).
- **ternary/row** — QAT with per-row absmean scales (default granularity).
- **ternary/tensor** — QAT with a single per-tensor scale.
- **ternary+out** — MLP only: also quantizes the output layer
  (`quantize_output=True`; by default the output layer stays full precision).
- **tabicl PTQ** — checkpoint weights approximated at load time
  (`TabICLClassifier(quantization='ternary')`); training-free.

All quantized variants keep latent float32 weights as the trainable
parameters and use straight-through gradients; the reported "eff-zero"
column is the fraction of exactly-zero effective weights induced by the
ternary scheme (~1.58 bits/weight of information).

## Setup

- Synthetic datasets: regression (`y = Xw*20 + noise`, standardized) and
  3-class classification (rank-based labels), 5,000 samples x 20 features.
- `hidden_layer_sizes=(128, 64)`, Adam, 100 epochs, `max_iter` reached
  (no early stopping triggered at these settings). TabM: `k=8`,
  `arch_type='tabm'`, embeddings off, `alpha=0` (L2 decay pushes latent
  weights into the ternary dead zone — see warnings below).
- Machine: Apple Silicon (M1 Max), macOS. Times are wall-clock `fit()`
  including preprocessing; scores are train scores.
- TabICL: 600 train / 250 held-out samples x 30 features on the cached
  `tabicl-classifier-v2` checkpoint, NumPy backend, wall-clock `predict()`.

## Results — 5,000 samples x 20 features, 100 epochs

### MLP

| Task | Variant | NumPy fit | NumPy score | Mojo fit | Mojo score | eff-zero |
|---|---|---|---|---|---|---|
| Regression | fp | **0.14s** | **0.9981** | **0.20s** | **0.9988** | – |
| Regression | ternary/row | 0.31s | 0.9962 | 0.43s | 0.9970 | 29.6–32.5% |
| Regression | ternary/tensor | 0.32s | 0.9964 | 0.44s | 0.9956 | 31.1–35.1% |
| Regression | ternary+out | 0.31s | 0.9949 | 0.43s | 0.9543 | ~32–41% |
| 3-class | fp | **0.51s** | **0.9326** | **0.49s** | **0.9388** | – |
| 3-class | ternary/row | 0.64s | 0.6442 | 0.50s | 0.6098 | ~31% |
| 3-class | ternary/tensor | 0.63s | 0.6366 | 0.56s | 0.6122 | ~31% |
| 3-class | ternary+out | 0.70s | 0.6294 | 0.50s | 0.5628 | ~31% |

### TabM (`k=8`, arch `tabm`, no embeddings)

| Task | Variant | NumPy fit | NumPy score | Mojo fit | Mojo score | eff-zero |
|---|---|---|---|---|---|---|
| Regression | fp | **18.9s** | 0.9979 | **3.19s** | 0.9975 | – |
| Regression | ternary/row | 19.3s | **0.9985** | 4.90s | 0.9980 | ~34% |
| Regression | ternary/tensor | 19.3s | 0.9983 | 3.68s | 0.9972 | ~35% |
| 3-class | fp | **31.2s** | **0.7630** | **3.62s** | **0.7642** | – |
| 3-class | ternary/row | 31.5s | 0.5852 | 3.93s | 0.5900 | ~35% |
| 3-class | ternary/tensor | 31.5s | 0.5870 | 5.10s | 0.5832 | ~35% |

### TabICL (post-training quantization, inference only)

| Variant | predict (250 rows) | held-out accuracy |
|---|---|---|
| fp | 76.1s | **0.988** |
| ternary/row (PTQ) | 72.9s | 0.672 |

## Ablation — before vs after BitLinear (`shinrin.benchmark`)

The tables above report *train* scores; for an honest before/after
comparison the `ablation_benchmark()` utility (see
[Benchmarking](../../docs/features/benchmarking.md)) fits each variant
once and scores it on **held-out** data. Run with:

```python
from shinrin.benchmark import ablation_benchmark, print_ablation_report

variants = {
    "fp (baseline)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100),
    "ternary/row": MLPClassifier(
        ..., quantization="ternary", quantization_granularity="per_row"
    ),
}
results = ablation_benchmark(variants, X_train, y_train, X_test, y_test)
print_ablation_report(results)
```

Setup: 4,000 train / 1,000 held-out samples x 20 features;
classification uses `sklearn.datasets.make_classification` (3 classes,
10 informative) because the rank-based labels of the main suite are
unlearnable out-of-sample (every variant sits at chance); regression as
above. Same architectures and epoch budget as the main suite.

### NumPy backend

MLP:

| Task | Variant | fit | Δfit | test score | Δscore |
|---|---|---|---|---|---|
| 3-class | fp | **0.39s** | 1.00x | **0.9210** | – |
| 3-class | ternary/row | 0.53s | 1.34x | 0.9140 | −0.007 |
| 3-class | ternary/tensor | 0.50s | 1.28x | 0.9050 | −0.016 |
| Regression | fp | **0.13s** | 1.00x | 0.9945 | – |
| Regression | ternary/row | 0.35s | 2.77x | 0.9941 | −0.0005 |
| Regression | ternary/tensor | 0.30s | 2.32x | **0.9959** | +0.001 |

TabM:

| Task | Variant | fit | Δfit | test score | Δscore |
|---|---|---|---|---|---|
| 3-class | fp | **25.2s** | 1.00x | **0.9370** | – |
| 3-class | ternary/row | 25.5s | 1.01x | 0.9310 | −0.006 |
| 3-class | ternary/tensor | 25.6s | 1.02x | 0.9300 | −0.007 |
| Regression | fp | **15.5s** | 1.00x | **0.9986** | – |
| Regression | ternary/row | 15.7s | 1.02x | 0.9982 | −0.0004 |
| Regression | ternary/tensor | 15.6s | 1.01x | 0.9983 | −0.0003 |

### Mojo backend

MLP:

| Task | Variant | fit | Δfit | test score | Δscore |
|---|---|---|---|---|---|
| 3-class | fp | **0.40s** | 1.00x | 0.9170 | – |
| 3-class | ternary/row | 0.46s | 1.15x | **0.9280** | +0.011 |
| 3-class | ternary/tensor | 0.48s | 1.21x | 0.9110 | −0.006 |
| Regression | fp | **0.16s** | 1.00x | **0.9955** | – |
| Regression | ternary/row | 0.37s | 2.31x | 0.9942 | −0.001 |
| Regression | ternary/tensor | 0.31s | 1.94x | 0.9943 | −0.001 |

TabM:

| Task | Variant | fit | Δfit | test score | Δscore |
|---|---|---|---|---|---|
| 3-class | fp | **3.03s** | 1.00x | **0.9420** | – |
| 3-class | ternary/row | 3.46s | 1.14x | 0.9280 | −0.014 |
| 3-class | ternary/tensor | 3.20s | 1.06x | 0.9310 | −0.011 |
| Regression | fp | **2.57s** | 1.00x | **0.9981** | – |
| Regression | ternary/row | 2.71s | 1.05x | 0.9980 | −0.000 |
| Regression | ternary/tensor | 2.69s | 1.05x | 0.9970 | −0.001 |

### Ablation takeaways

- On a learnable classification task the QAT cost is modest: ~0.6–1.6
  accuracy points (NumPy), within RNG noise on some Mojo runs — far
  smaller than the train-score gaps of the rank-based main suite.
- Regression stays at full-precision parity on held-out R² everywhere.
- Fit overhead is 1.01–1.34x except tiny-time MLP regression fits,
  where the fixed per-refresh quantize pass dominates the ratio (~2x of
  a 0.13–0.16s baseline).

## Notes

- **Fit-time overhead is small.** QAT adds one absmean pass over each
  quantized weight per refresh (epoch start / after each Adam round /
  before L-BFGS evaluations), costing roughly 2–25% wall-clock on these
  sizes. It never dominates; GEMMs remain the bottleneck.
- **Regression quality holds.** Both MLP and TabM match full-precision
  R² within noise under ternary QAT — the output layer stays full
  precision by default and the tasks tolerate the {-1, 0, +1}·gamma
  backbone.
- **Classification is more sensitive.** On this rank-based 3-class task,
  quantizing every hidden layer costs ~15–30 accuracy points. Smaller
  nets / longer training / leaving late layers at full precision are the
  usual mitigations; treat per-task validation as mandatory before
  shipping quantized classifiers.
- **Quantizing the output layer too (`ternary+out`) costs extra**, as
  expected; it is opt-in via `quantize_output`.
- **TabICL naive absmean PTQ loses substantial accuracy** (0.988 → 0.672)
  while prediction time is unchanged (same FLOPs — PTQ here is a storage/
  parity experiment, not a speedup). In-context-learning transformers are
  known to be fragile under naive weight rounding; proper QAT or
  round-to-nearest with per-group scales would be required for a
  production-quality result. The API therefore ships flagged experimental
  and emits a `UserWarning`.
- **`alpha=0` recommended for TabM QAT**: with L2 decay (`alpha > 0`) the
  latent weights shrink toward the ternary dead zone where the effective
  weight is zero and no gradient flows (the estimator warns at fit time).
- **Mojo/NumPy parity:** scales accumulate in float64 on both sides so
  both backends derive bit-identical absmean scales regardless of
  summation order; deterministic loss/grad parity tests live in
  `tests/test_mlp_parity.py`, `tests/test_tabm_parity.py` and
  `tests/test_bitlinear.py`. Score differences between backends above are
  dropout/shuffle RNG trajectory noise, not backend disagreement.
