# GOSDT Benchmarks

Comparison of shinrin's vendored **GOSDTClassifier** (globally optimal sparse
decision trees with reference-ensemble guesses) against a typical decision
tree classifier, scikit-learn's CART (`DecisionTreeClassifier`):

- **cart** — CART on raw features, unlimited depth (sklearn defaults,
  `random_state=0`).
- **cart+d** — CART with `max_depth` set to GOSDT's depth budget
  (matched-complexity reference).
- **gosdt** — full reference-ensemble pipeline: `ThresholdGuessBinarizer`
  (n_estimators=20, max_depth=2) + `GOSDTClassifier`
  (regularization=0.05, depth_budget=4). Binarization time is reported
  separately; datasets that are already binary (compas) skip it.

Every number is a single fit on a stratified 75/25 train/test split
(`random_state=0`). "cert gap" is GOSDT's certified optimality interval
`upper_bound - lower_bound`; 0.0000 means the returned tree is provably
optimal for its objective.

To run: `uv run python scripts/benchmarks/bench_gosdt.py [--repeats N]
[--regularization F] [--depth-budget N]` (or `just bench-gosdt`).

## Setup

- GOSDT runs on shinrin's vendored engine, which executes **single-threaded**
  (serial TBB shim) — upstream's multithreaded oneTBB build would be faster.
- The binarizer uses smaller settings than upstream defaults (20 estimators x
  depth 2 instead of 100 x depth 3): with 100x3, threshold-column explosion
  makes both column elimination and the single-threaded search impractical.
- Machine: Apple Silicon M1 Max (arm64), macOS, CPython 3.14, clang -O3.

## Results

| Dataset | Model | Binarize | Fit | Predict (ms) | Test acc | Nodes | Leaves | Cert gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iris (n=150, d=4, 3-class) | cart | – | 0.001s | 0.07 | 0.9737 | 15 | 8 | – |
| | cart+d | – | 0.000s | 0.06 | 0.9737 | 15 | 8 | – |
| | gosdt | 0.101s | 0.001s | 0.08 | 0.9737 | **5** | **3** | 0.0000 |
| small synthetic (n=2000, d=10) | cart | – | 0.006s | 0.17 | **0.8720** | 233 | 117 | – |
| | cart+d | – | 0.004s | 0.11 | 0.8700 | 31 | 16 | – |
| | gosdt | 0.146s | 0.041s | 0.36 | 0.8280 | **5** | **3** | 0.0000 |
| medium synthetic (n=10000, d=20) | cart | – | 0.094s | 0.37 | **0.8932** | 857 | 429 | – |
| | cart+d | – | 0.039s | 0.23 | 0.8428 | 31 | 16 | – |
| | gosdt | 0.981s | 2.55s | 1.27 | 0.7992 | **5** | **3** | 0.0000 |
| compas (real, binary, n=7214, d=27) | cart | – | 0.005s | 0.31 | 0.6519 | 1187 | 594 | – |
| | cart+d | – | 0.003s | 0.19 | **0.6752** | 31 | 16 | – |
| | gosdt | – | 0.010s | 0.62 | 0.6314 | **3** | **2** | 0.0000 |

### Cost of tighter regularization (medium synthetic, depth_budget=5)

| regularization | fit time | test acc | cert gap |
|---:|---:|---:|---:|
| 0.05 / depth 4 | 2.6s | 0.7992 | 0.0000 |
| 0.02 / depth 5 | 204.9s | 0.8164 | 0.0000 |
| 0.01 / depth 5 | 435.6s | 0.8164 | 0.0000 |

## Takeaways

- **Speed**: CART is 10–1000x faster to fit, as expected for a greedy
  heuristic. GOSDT's end-to-end pipeline stays practical — sub-second on
  real binary data (compas: 10ms), ~3.5s including binarization on
  10k x 20 continuous features at depth 4.
- **Model size is the headline**: GOSDT returns 2–5 leaf trees where CART
  grows 8–594 leaves. On compas that is a 2-split rule list vs a 594-leaf
  tree, at 0.63 vs 0.68 held-out accuracy — a very different operating point
  on the interpretability axis, backed by a certificate of optimality
  (gap 0.0000 in every configuration tested).
- **Accuracy**: unlimited-depth CART scores highest on noisy synthetic sets;
  note however that plain depth-limiting (`cart+d`) recovers most of CART's
  accuracy while GOSDT trades more accuracy for far fewer leaves at this
  regularization. Lowering regularization closes the accuracy gap at steep
  computational cost (~80x slower from 0.05/d4 to 0.02/d5 for +1.7pp here).
- **Prediction** costs are comparable across models; GOSDT's tiny trees make
  inference effectively free despite the extra Python-level tree walk.
- Runtime is dominated by the search's branch-and-bound; the single-threaded
  vendored build is the main overhead versus upstream's parallel builds.
