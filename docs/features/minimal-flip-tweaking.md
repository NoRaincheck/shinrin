# Minimal-Flip Feature Tweaking

Given a fitted tree model, find the **smallest change to a sample's features
that flips the model's prediction** — actionable counterfactuals in the sense
of Tolomei et al., *Interpretable Predictions of Tree-based Ensembles via
Actionable Feature Tweaking* (KDD 2017,
[arXiv:1706.06691](https://arxiv.org/abs/1706.06691)), reimplemented for
shinrin's sparse-tree stack.

The shinrin variation: instead of tweaking against one arbitrary member of an
ensemble, you can ask for the minimal tweak that flips **every member of a
Rashomon set simultaneously** — a recommendation guaranteed to hold no matter
which near-optimal tree is deployed.

## Scopes

| Scope | Flips | Guarantee |
|---|---|---|
| `scope="reference"` | The first/optimal tree only | Exact minimum for that single tree |
| `scope="rashomon"` | Every model of the set | Exact minimum over all-members-flip tweaks; emptied search certifies infeasibility |
| `scope="ensemble"` | The estimator's own aggregated prediction | Verified against the real `predict` |

!!! note "Guarantees are per individual tree"

    Minimality and the verified flip hold on a single-tree basis. Tweaking one
    member of an ensemble says nothing about the aggregated vote of the rest —
    single-tree tweaking is *not* robust in ensembles. That makes it ideal for
    [SPOT](../models/spot.md), where the whole model is one globally optimal
    sparse tree. Robustness across many models is available via `rashomon`
    (cheap on SPOTSETs, typically infeasible on decorrelated forests) or
    `ensemble` (aggregate-level, weaker than all-members robustness).

## Supported estimators

Any fitted classifier whose decisions decompose into root-to-leaf per-feature
tests:

- `shinrin.SPOTClassifier` and `shinrin.SPOTSETClassifier`
- scikit-learn decision trees, forests, bagging ensembles (soft probability votes)
- `AdaBoostClassifier` (weighted hard votes)
- binary `GradientBoostingClassifier` / `HistGradientBoostingClassifier`
  (score-sign crossing over stage outputs)

Tweaks are reported in each estimator's training space: binarized columns for
SPOT/SPOTSET, raw features for scikit-learn models. Distances compare within a
family only (Hamming vs L1).

## Quick start: SPOTSET robust tweaks

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from shinrin import (
    RashomonFlipSearch,
    SPOTSETClassifier,
    ThresholdGuessBinarizer,
    summarize_flip_results,
)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
Xb_tr = enc.fit_transform(X_tr[:400], y_tr[:400]).astype(float)
Xb_te = enc.transform(X_te).astype(float)

spotset = SPOTSETClassifier(
    regularization=0.01, rashomon_bound_multiplier=0.1, depth_budget=3
).fit(Xb_tr, y_tr[:400])
print(spotset.n_trees_)                       # 16 trees in the set

search = RashomonFlipSearch(spotset)
negatives = np.flatnonzero(spotset.predict(Xb_te) == 0)[:5]

reference = search.search(Xb_te[negatives], target=1, scope="reference")
robust = search.search(Xb_te[negatives], target=1, scope="rashomon")

r = robust[2]
print(r.changed_features)                     # binarized columns to toggle
print(r.l1_distance, r.optimal, r.verified)   # cost, optimality proof, re-check
```

Output:

```
16
(11, 19, 20)
3.0 True True
```

Sample 2 needs **one** column toggled to flip the first (optimal) tree, but
**three** so that all 16 near-optimal trees flip together. Every reported
tweak is re-verified by predicting with all members; a solver/verification
mismatch raises instead of returning silently.

## Batch summaries

```python
from pprint import pprint
pprint(summarize_flip_results(reference))
pprint(summarize_flip_results(robust))
```

Output:

```
{'budget_exhausted_rate': 0.0,
 'max_distance': 2.0,
 'mean_changed_features': 1.8,
 'mean_distance': 1.8,
 'median_distance': 2.0,
 'n_samples': 5,
 'proven_infeasible_rate': 0.0,
 'success_rate': 1.0,
 'total_nodes': 5,
 'total_time_s': 1.06e-05}
```

```
{'budget_exhausted_rate': 0.0,
 'max_distance': 6.0,
 'mean_changed_features': 5.2,
 'mean_distance': 5.2,
 'median_distance': 6.0,
 'n_samples': 5,
 'proven_infeasible_rate': 0.0,
 'success_rate': 1.0,
 'total_nodes': 80,
 'total_time_s': 0.000327}
```

The robust tweak costs more than the single-tree one — the price of a
guarantee that holds across the entire Rashomon set rather than an arbitrary
member of it — yet stays small because near-optimal trees share most of their
decision structure.

## Random forests and boosted ensembles

`scope="ensemble"` works on any supported scikit-learn tree model and uses its
true decision rule: soft probability voting for forests/trees/bagging,
AdaBoost's weighted hard votes, score-sign crossing for binary boosting
ensembles.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=9, max_depth=4, random_state=0
).fit(X_tr[:400], y_tr[:400])

results = RashomonFlipSearch(rf).search(X_te[negatives], target=1, scope="ensemble")
print(results[1].l1_distance, results[1].changed_features, results[1].verified)
```

Output:

```
0.7010250195813179 (10, 17, 26, 27, 29) True
```

Large forests remain valid targets, but hard queries may exhaust budgets:
`FlipResult.optimal=False` then means "best found, optimality unproven"
(tune with `max_nodes=` / `time_limit=`), while `success=False, optimal=True`
certifies that no flipping tweak exists.

## Benchmark highlights

Full run and methodology:
[benchmarks/RASHOMON_FLIP_BENCHMARK.md](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/RASHOMON_FLIP_BENCHMARK.md)
(`just bench-rashomon-flip`). Reproduce locally with
`uv run python scripts/benchmarks/bench_rashomon_flip.py --smoke`.

| Config (40 samples each) | ok | mean d | nodes |
|---|---:|---:|---:|
| spotset/ref (197 trees, compas) | 100% | 1.00 | 80 |
| spotset/rashomon | 100% | 4.63 | 30,908 |
| rf64/ref | 100% | 0.53 | 1,200 |
| rf64/rashomon (breast cancer) | 10% | – (90% budget-exhausted) | 8.6M |

Flipping every member of a 197-tree Rashomon set solves in sub-second time;
the identical all-trees query on a 64-tree random forest exhausts 500k-node /
10 s budgets on 90% of continuous-feature samples — decorrelation makes
all-trees flips effectively unsolvable at realistic budgets.

## API

### `RashomonFlipSearch(estimator)`

Wraps a *fitted* tree-like classifier. Raises `TypeError` for unsupported
estimators and `NotImplementedError` for multiclass boosted models.

### `RashomonFlipSearch.search(X, *, target=None, scope="rashomon", max_nodes=100_000, time_limit=None) -> list[FlipResult]`

| Parameter | Description |
|---|---|
| `X` | Feature matrix in the estimator's training space (binarized for SPOT/SPOTSET) |
| `target` | Desired class in original label space; `None` = any other class (binary only) |
| `scope` | `"reference"`, `"rashomon"` or `"ensemble"` |
| `max_nodes` | Search-node budget per sample |
| `time_limit` | Per-sample wall-clock budget in seconds |

### `FlipResult` fields

| Field | Description |
|---|---|
| `success` | A flipping tweak was found |
| `optimal` | Cost is provably minimal (`False` when budgets were exhausted); `success=False, optimal=True` certifies infeasibility |
| `verified` | Tweak re-checked by predicting with every model / the estimator itself |
| `x_new` | The tweaked feature vector (`None` if unsuccessful) |
| `changed_features` | Column indices moved by the tweak |
| `l1_distance` | L1 distance moved (Hamming in binarized space) |
| `n_models_total` / `n_models_agree_before` / `n_models_flipped_after` | Set size, members already predicting the target before, members supporting it after |
| `nodes_expanded`, `time_s` | Solver effort |

### `summarize_flip_results(results) -> dict`

Aggregates success/infeasibility rates, mean/median/max distances, changed
features, and total solver effort over a batch.
