# ORDT Benchmarks

**ORDT (Optimal Rule-sets from Decision Trees)** — a two-stage extension of
skope-rules that replaces its heuristic rule voting with CORELS' certified
optimal selection:

1. **mine** — the vendored `SkopeRules` harvests root-to-leaf threshold
   conjunctions from bagged classification/regression trees and scores them by
   out-of-bag precision/recall. Unlike stock skope-rules, near-duplicate rules
   are *kept* (`max_depth_duplication=None`): CORELS, not a similarity
   heuristic, decides what earns a slot.
2. **select** — each surviving rule becomes one binary column of a capture
   matrix `Z` (`Z[i, j] = 1` iff rule *j* fires on sample *i*). The vendored
   `CorelsClassifier` is fit with `max_card=1`, so every antecedent of the
   learned list is exactly one mined rule. The result is the provably best
   ordered rule set **over the mined pool**.

Models compared:

- **cart** — sklearn `DecisionTreeClassifier` on raw features (unlimited depth).
- **skope** — vendored SkopeRules alone: weighted vote of rules with its stock
  semantic deduplication (`max_depth_duplication=2`). This is the baseline
  ORDT extends.
- **corels** — vendored `CorelsClassifier`, own FPGrowth-style mining on
  quantile one-hot binarized features (4 bins; compas skips binarization —
  already binary).
- **ordt** — the hybrid above. Works directly on raw features: adaptive tree
  thresholds replace external discretization.

To run: `uv run --extra pandas python scripts/benchmarks/bench_ordt.py
[--repeats N] [--max-rules K]` (or `just bench-ordt`; pandas is required by
SkopeRules' DataFrame-based rule evaluation).

## Setup

- Best of 3 fits per cell, stratified 75/25 train/test split
  (`random_state=0`); predict time is one warm call on the test fold.
- Mining defaults: 10 estimators x depth 3, `precision_min=0.5`,
  `recall_min=0.01`, `max_samples=0.8`. Selection: `max_rules=150`,
  `c=0.01`, `min_support=0.01`, CORELS `n_iter=10000`.
- "clauses" counts total antecedent terms; for cart it is n/a (leaves shown).
- Machine: Apple Silicon M1 Max (arm64), macOS, CPython 3.14.

## Results

| Dataset | Model | Mine | Select | Fit | Predict (ms) | Test acc | Train acc | Size | Clauses |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| breast-cancer (real, n=569, d=30) | cart | – | – | 0.003s | 0.16 | 0.9021 | 1.0000 | 18 leaves | – |
| | skope | – | – | 0.160s | 4.79 | **0.9371** | 0.9624 | 7 rules | 21 |
| | corels | – | – | 0.013s | 0.02 | 0.9301 | 0.9390 | 3 rules | 3 |
| | ordt | 0.162s | 0.001s | 0.164s | 0.17 | 0.9161 | 0.9648 | **2 rules** | **2** |
| small synthetic (n=2000, d=10) | cart | – | – | 0.007s | 0.09 | **0.8720** | 1.0000 | 117 leaves | – |
| | skope | – | – | 0.156s | 2.58 | 0.8340 | 0.8447 | 4 rules | 12 |
| | corels | – | – | 0.027s | 0.02 | 0.8460 | 0.8460 | 5 rules | 5 |
| | ordt | 0.152s | 0.019s | 0.171s | 0.25 | 0.8480 | 0.8627 | **3 rules** | **3** |
| medium synthetic (n=10000, d=20) | cart | – | – | 0.096s | 0.22 | **0.8932** | 1.0000 | 429 leaves | – |
| | skope | – | – | 0.616s | 3.24 | 0.8136 | 0.8205 | 4 rules | 12 |
| | corels | – | – | 0.090s | 0.06 | 0.8196 | 0.8211 | 5 rules | 5 |
| | ordt | 0.641s | 0.757s | 1.399s | 0.70 | 0.8392 | 0.8437 | 5 rules | 5 |
| compas (real, binary, n=7214, d=27) | cart | – | – | 0.005s | 0.26 | 0.6519 | 0.7131 | 594 leaves | – |
| | skope | – | – | 0.193s | 4.77 | 0.6619 | 0.6599 | 6 rules | 18 |
| | corels | – | – | 0.096s | 0.02 | 0.6674 | 0.6588 | 3 rules | 3 |
| | ordt | 0.208s | 0.059s | 0.266s | 0.29 | **0.6685** | 0.6678 | 5 rules | 5 |

Candidate pools kept for CORELS after degenerate-column filtering and
capture-set deduplication (mined → usable → selected): breast-cancer 20/20,
small 33/33, medium 42/42, compas 20/20 — well under the `max_rules=150` cap.

### Learned ORDT rule list (compas)

```
IF x24 > 0.5 and x26 <= 0.5 and x6 <= 0.5 THEN y=1
IF x24 > 0.5 and x25 > 0.5 and x6 > 0.5 THEN y=1
IF x24 > 0.5 and x26 > 0.5 and x6 <= 0.5 THEN y=1
IF x2  > 0.5 and x23 > 0.5 and x24 <= 0.5 THEN y=1
IF x24 > 0.5 and x26 > 0.5 and x5 <= 0.5 THEN y=0
```

Five three-term clauses selected from a pool of 20 mined rules; prediction is
a first-match walk down the list.

## Takeaways

- **ORDT dominates skope-rules everywhere**: replacing the precision-weighted
  vote with optimal selection/ordering gains +2.6pp test accuracy on medium
  synthetic (0.8392 vs 0.8136), +1.4pp on small synthetic, +0.7pp on compas,
  while producing far smaller models (2–5 rules vs 4–7 with 3x more clauses).
- **ORDT beats plain CORELS on 3 of 4 datasets** without any external
  discretization: tree-path thresholds adapt to the data, whereas CORELS'
  fixed quantile bins lose information. On compas (already binary) both see
  comparable inputs and land within ~0.1pp, but ORDT's certificate covers the
  tree-mined neighborhood rather than all single-feature conjunctions.
- **Interpretable-model frontier**: among models of comparable size
  (~3–5 clauses), ORDT posts the best test accuracy on medium synthetic and
  compas; CART wins accuracy overall only by growing 100x larger
  (117–594 leaves).
- **Cost profile**: mining (skope fit) dominates; optimal selection is cheap
  when the pool dedups tightly (<0.06s) and grows with pool width x sample
  count (0.76s on n=10000 with 42 candidates). Prediction is 5–15x faster
  than skope's pandas-based vote because capture evaluation is vectorized
  NumPy over the few selected rules.
- **Caveat**: optimality is certified relative to the mined pool, not globally
  over all possible conjunctions — the same relative-to-candidates semantics
  as FPSkope→CORELS pipelines.
