# CORELS Optimal Rule Lists

CORELS (Certifiably Optimal RulE ListS) learns a *provably optimal* rule list
— a series of if/then statements — over binary features via branch-and-bound
with aggressive pruning bounds. Where CART grows a tree greedily top-down,
CORELS certifies that no rule list with a better regularized objective exists.

Shinrin vendors [pycorels](https://github.com/corels/pycorels) (GPL-3.0) and
compiles its C++ engine into the `shinrin._native` extension with **bundled
mini-GMP**, so there is no libgmp system dependency and wheels are fully
self-contained. See NOTICE for attribution details.

> **Note:** unlike most of shinrin, this estimator is derived from GPL-3.0
> code. The license text ships alongside the vendored sources.

## CorelsClassifier

```python
from shinrin import CorelsClassifier

# X must be binary (0/1) features; y binary labels
clf = CorelsClassifier(c=0.01, verbosity=["rulelist"])
clf.fit(X, y, features=["Age<=25", "Prior-Crimes>3"])

print(clf.rl())                 # human-readable optimal rule list
accuracy = clf.score(X, y)
predictions = clf.predict(X)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `c` | `float` | `0.01` | Per-rule-list-node penalty. Higher values prefer shorter lists |
| `n_iter` | `int` | `10000` | Maximum number of search nodes explored |
| `map_type` | `str` | `"prefix"` | Prefix map: `"none"`, `"prefix"` or `"captured"` |
| `policy` | `str` | `"lower_bound"` | Node ordering: `"bfs"`, `"curious"`, `"lower_bound"`, `"objective"` or `"dfs"` |
| `verbosity` | `list[str]` | `["rulelist"]` | Subset of `"rulelist"`, `"rule"`, `"label"`, `"minor"`, `"samples"`, `"progress"`, `"mine"`, `"loud"` (`[]` is silent) |
| `ablation` | `int` | `0` | Bound configuration: 0 all bounds, 1 no antecedent support bound, 2 no lookahead bound |
| `max_card` | `int` | `2` | Max conjunction size when mining rules |
| `min_support` | `float` | `0.01` | Fraction of samples a mined rule must capture (0–0.5) |

### Attributes / methods

| Member | Description |
|---|---|
| `fit(X, y, features=[], prediction_name="prediction")` | Fit; returns `self`. Warnings suggest valid `c` ranges for the sample count |
| `rl_` / `rl()` | Learned [`RuleList`](#rulelist) |
| `predict(X)` / `predict_proba`-free `score(X, y)` | Predictions and accuracy on binary data |
| `save(fname)` / `load(fname)` | Pickle round-trip of the fitted model |

## RuleList

The learned model: `rules` (each a dict of `antecedents` feature indices,
negatives encoded as `-i`, plus a boolean `prediction`),
`features` (names) and `prediction_name`. Its `__str__` renders the full
if/then list:

```
RULELIST:
if [Prior-Crimes>5]:
  Recidivate-Within-Two-Years = True
else if [Age<=40]:
  Recidivate-Within-Two-Years = False
else 
  Recidivate-Within-Two-Years = True
```

## load_from_csv

Load binary CSV datasets in the format CORELS expects (header row with
feature names, final column is the label):

```python
from shinrin._corels import load_from_csv

X, y, features, prediction_name = load_from_csv("compas.csv")
```

## OrdtClassifier

`OrdtClassifier` is a variant of shinrin's vendored
[SkopeRules](https://github.com/scikit-learn-contrib/skope-rules)
(`shinrin[pandas]` required) that replaces skope's heuristic
precision-weighted vote with CORELS' certified-optimal selection:

1. **mine** — skope-rules harvests root-to-leaf threshold conjunctions from
   bagged classification/regression trees, scored by out-of-bag
   precision/recall. Near-duplicates are deliberately kept: CORELS, not a
   similarity heuristic, decides what earns a slot.
2. **select** — every surviving rule becomes one binary column of a capture
   matrix (`Z[i, j] = 1` iff rule *j* fires on sample *i*), and
   `CorelsClassifier(max_card=1)` returns the provably best ordered rule list
   *over the mined pool*. Tree thresholds work directly on raw features — no
   external binarization needed.

```python
from shinrin import OrdtClassifier

clf = OrdtClassifier(n_estimators=10, max_depth=3, random_state=0)
clf.fit(X, y)                  # binary targets {0, 1}
accuracy = clf.score(X_test, y_test)

clf.list_rules()               # ordered (rule label, prediction) pairs;
                               # negated candidates are prefixed with "NOT "
clauses, n_rules = clf.complexity()
clf.stats_                     # {"mined": ..., "usable": ..., "selected": ...}
```

Mining parameters are inherited from `SkopeRules`; `max_rules`, `c`,
`min_support` and `n_iter` configure the selection stage.

Optimality is certified relative to the mined candidate pool, not globally
over all possible conjunctions. Measured in
[ORDT_BENCHMARK](https://github.com/NoRaincheck/shinrin/blob/main/scripts/benchmarks/ORDT_BENCHMARK.md)
(`just bench-ordt`), this hybrid beats skope-rules alone on every dataset
tested — up to +2.6pp test accuracy with 2–5-clause lists — and beats
CORELS' own mining on 3 of 4 datasets, since adaptive tree thresholds
replace fixed quantile bins.

## Notes and constraints

- Features must already be binary; binarize continuous data yourself (or use
  [`ThresholdGuessBinarizer`](spot.md#thresholdguessbinarizer) from the SPOT
  stack).
- Binary classification only.
- Fitting uses module-global state in the native engine, matching upstream;
  fits are not thread-safe but may be interleaved with other work.
- Benchmarks vs scikit-learn CART: see
  [Benchmarking](../features/benchmarking.md#scripts).
