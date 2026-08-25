# Rashomon Minimal-Flip Benchmark

Minimal **feature tweaking** (Tolomei et al., *Interpretable Predictions of
Tree-based Ensembles via Actionable Feature Tweaking*, KDD 2017,
[arXiv:1706.06691](https://arxiv.org/abs/1706.06691)) reimplemented for
shinrin's sparse-tree stack, with one key variation: instead of computing a
tweak against a single reference model, search for the minimal tweak that
flips **every member of a model set simultaneously**:

- `spot/ref` — single optimal sparse tree (SPOT); classical counterfactual.
- `spotset/ref` — first (lowest-objective) tree of the SPOTSET Rashomon set;
  what most counterfactual tooling would report.
- `spotset/rashomon` — **every tree of the Rashomon set at once**: a tweak
  guaranteed to flip the prediction regardless of which near-optimal model
  is deployed.
- `rf{k}/ref`, `rf{k}/rashomon` — same two queries against a k-tree random
  forest, where "the model" is ambiguous by construction.

Motivating question: Rashomon members are near-optimal solutions to the same
objective and therefore share most decision structure. Does that make a
robust all-models tweak cheap to find, while the identical query on a
decorrelated random forest becomes hard or unsolvable?

## Method

Each tree reduces to root-to-leaf literal sets: a leaf becomes per-feature
intervals (`x_f == v` -> `[v, v]`; `x[f] <= t` -> `[-inf, t]`; `x[f] > t` ->
a float32-safe successor bound, because sklearn compares in float32).
Flipping all M models reduces to choosing one opposite-class leaf per model
whose literal sets have a non-empty intersection; the tweak is the L1
projection of `x` onto that intersection, so its cost is exact for that
combination. The joint search is A* over partial combinations with an
admissible suffix lower bound, canonical-state dedup with reopening, and
node/time budgets; budget exhaustion yields "best found, optimality
unproven", while an emptied queue certifies infeasibility.

## Setup

- Datasets: breast cancer (sklearn, n=569, d=30, continuous; binarized with
  `ThresholdGuessBinarizer` 20x depth 2) and compas (n=7214, d=27, already
  binary). Stratified 75/25 split, seed 0; sparse pipeline trained on the
  first 400 training rows.
- SPOT: regularization 0.01, depth budget 3. SPOTSET: same plus
  `rashomon_bound_multiplier=0.1` (16 trees on breast cancer, 197 on
  compas). RF: `RandomForestClassifier(n_estimators=64, max_depth=8)`.
- Query samples: 40 test rows per family whose reference model predicts the
  negative class; requested flip is towards class 1.
- Budgets: 500k search nodes, 10 s/sample (rashomon scope only).
- Distances are Hamming in binarized space for spot/spotset and L1 in raw
  feature space for forests — comparable within a family only.
- Every reported tweak is re-verified by predicting with every model.

To run: `uv run python scripts/benchmarks/bench_rashomon_flip.py`
(or `just bench-rashomon-flip`); `--smoke` for a quick pass. Machine:
Apple Silicon M1 Max, CPython 3.14.

## Results

`ok` = robust tweak found (verified on all models), `infeas` = search
provably proved no all-models tweak exists, `budget` = limits exhausted
without a proof. `d` = mean distance over solved samples only.

| Dataset | Config | ok | infeas | budget | d (ref vs robust) | feats | nodes | time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| breast-cancer (16-tree set) | spot/ref | 100% | 0% | 0% | 1.73 | 1.73 | 80 | 0.001s |
| | spotset/ref | 100% | 0% | 0% | 1.63 | 1.63 | 40 | 0.002s |
| | spotset/rashomon | **100%** | 0% | 0% | **4.60** | 4.60 | **640** | **0.006s** |
| | rf64/ref | 100% | 0% | 0% | 3.34 | 1.70 | 320 | 0.119s |
| | rf64/rashomon | **15%** | 0% | **85%** | 6.89 | 7.67 | 8,409,820 | 367s |
| compas (197-tree set) | spot/ref | 100% | 0% | 0% | 1.00 | 1.00 | 120 | 0.001s |
| | spotset/ref | 100% | 0% | 0% | 1.00 | 1.00 | 80 | 0.024s |
| | spotset/rashomon | **100%** | 0% | 0% | **6.33** | 6.33 | **9,466** | **0.141s** |
| | rf64/ref | 100% | 0% | 0% | 0.53 | 1.05 | 1,200 | 0.189s |
| | rf64/rashomon | 100% | 0% | 0% | 2.86 | 5.72 | 216,296 | 8.347s |

## Findings

1. **The Rashomon-robust tweak is cheap and always found.** Flipping every
   member of a 16- or 197-tree Rashomon set succeeds for 100% of samples in
   milliseconds (640–9.5k A* nodes total). The shared structure hypothesis
   holds: near-optimal trees disagree on few paths, so one small
   intersection of leaf constraints flips them all at once.
2. **Robustness costs little on sparse trees but is real.** Mean distance
   grows from ~1.6 to 4.6 (breast cancer) and 1.0 to 6.3 (compas) toggled
   binarized features when moving from the single optimal tree to the whole
   set — the price of a guarantee that holds across the entire Rashomon
   set rather than an arbitrary member of it.
3. **Random forests struggle with the identical query.** On breast cancer,
   rf64/rashomon solved only 15% of samples within budget (85% unresolved
   after 500k nodes / 10 s each; 8.4M nodes burned overall), versus 100%
   success for spotset/rashomon in under a millisecond per sample. The RF's
   trees are decorrelated by design — their opposite-class leaf regions
   rarely intersect nicely, and the joint constraint system becomes
   effectively unsolvable at realistic budgets.
4. **Binary-feature data softens the forest case** (compas: RF solved, d
   0.53 -> 2.86, high pre-tweak agreement 0.43). Difficulty tracks feature
   richness and tree diversity rather than tree count alone.
5. Caveat: rf64/rashomon "d" averages solved samples only (15%), so it
   understates difficulty; the honest signal there is the budget-exhaustion
   rate.

## API

```python
from shinrin import RashomonFlipSearch

search = RashomonFlipSearch(fitted_spotset_or_forest)
results = search.search(X, target=1, scope="rashomon")   # or "reference"
summary = summarize_flip_results(results)
```

`FlipResult` exposes `x_new`, `changed_features`, `l1_distance`,
`optimal`, `verified`, and per-model agreement counts. Every successful
tweak is re-verified by predicting with all models; a solver/verification
mismatch raises instead of returning silently.
