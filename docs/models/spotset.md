# SPOTSET Rashomon Sets of Sparse Trees (formerly treeFARMS)

SPOTSET (**S**parse **O**ptimal **R**ashomon **SET**) enumerates the *whole
Rashomon set* of sparse decision trees: instead of returning one optimal tree,
it returns every tree whose regularized objective (misclassification loss plus
a per-leaf penalty) lies within a configurable bound of the optimum.

> Renamed from treeFARMS ("Trees FAst RashoMon Sets",
> [ubc-systopia/treeFARMS](https://github.com/ubc-systopia/treeFARMS),
> NeurIPS 2022). Upstream builds on the same gosdt-guesses lineage as
> [SPOT](spot.md) and names nearly everything "GOSDT"; in shinrin the engine is
> renamed SPOTSET to keep the single-optimal-tree trainer (`SPOTClassifier`,
> formerly `GOSDTClassifier`) distinct from the set-enumerating trainer. The
> two engines are compiled into the same native extension (the SPOTSET engine
> lives behind a `spotset` C++ namespace), so there is no duplicated binary or
> system dependency.

## Why a Rashomon set?

Many datasets admit several almost-equally-good trees with *very different*
structure. The Rashomon set exposes that ambiguity: you can inspect competing
explanations, select for robustness/simplicity trade-offs after training, and
quantify how much accuracy must be sacrificed for sparser models (see McTavish
et al., *Exploring the Whole Rashomon Set of Sparse Decision Trees*, NeurIPS
2022).

## End-to-end pipeline

```python
from shinrin import SPOTSETClassifier, ThresholdGuessBinarizer

# 1) Binarize via reference-ensemble threshold guesses (same as SPOT)
enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
X_bin = enc.fit_transform(X, y)

# 2) Enumerate all trees within 5% of the optimal regularized objective
clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.05)
clf.fit(X_bin, y)

print(clf.n_trees_)          # number of trees in the extracted set
accuracy = clf.score(X_bin, y)   # first tree's accuracy (sklearn convention)
```

## Working with the set

```python
tree = clf[1]                     # decode the second tree of the set
print(tree.leaves())              # number of leaves
print(tree.maximum_depth())       # longest decision path
print(tree)                       # if-then-else pseudocode
y_hat = tree.predict(X_bin)       # predictions (integer-coded labels)

metric = clf.model_set_.get_tree_metric_at_idx(1)
# {"objective": ..., "loss": ..., "complexity": ...} for that tree

trie = clf.get_decision_paths()   # trie of shared decision paths across the set
```

Notes:

- `clf.predict` / `clf.score` use the **first** tree of the set (the
  lowest-objective model), mapping integer-coded labels back to
  `clf.classes_`. Trees obtained through `clf[i]` predict integer-coded
  labels; map back with `clf.classes_[prediction]`.
- Every tree's objective is guaranteed within
  `(1 + rashomon_bound_multiplier)` times the optimum — verified against
  SPOT in `tests/test_spotset.py`.

## SPOTSETClassifier

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `regularization` | `float` | `0.05` | Penalty per leaf; recommend > `1 / n_samples` |
| `rashomon_bound_multiplier` | `float` | `0.05` | Set size knob: bound = `(1 + multiplier) * optimum`; grows the set exponentially |
| `rashomon` | `bool` | `True` | Extract the full set; `False` returns only the near-optimal search result |
| `depth_budget` | `int \| None` | `None` | Max tree depth (root-only tree = depth 1); `None` is unlimited |
| `time_limit` | `int \| None` | `None` | Seconds; on timeout the partial set is returned with a warning |
| `worker_limit` | `int` | `1` | Parallel search workers; `0` uses one per core |
| `verbose` | `bool` | `False` | Engine progress printing |

### Attributes

| Attribute | Description |
|---|---|
| `classes_` | Unique class labels seen during fit |
| `model_set_` | `ModelSetContainer` over the extracted set |
| `n_trees_` | Number of trees in the set |
| `train_time_` | Native search time in seconds |
| `n_features_in_` | Number of input features |

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Extract the Rashomon set for binarized features |
| `predict(X)` | First-tree predictions mapped to original labels |
| `score(X, y)` | First-tree accuracy |
| `get_tree_count()` | Size of the set |
| `__getitem__(i)` | Decode the i-th tree as a `TreeClassifier` |
| `get_decision_paths()` | Trie of decision paths shared across the set |

## Provenance

Vendored from [ubc-systopia/treeFARMS](https://github.com/ubc-systopia/treeFARMS)
(BSD-3-Clause). Deviations from upstream are documented in
[`src/shinrin/_spotset/README.md`](https://github.com/NoRaincheck/shinrin/blob/main/src/shinrin/_spotset/README.md):
the timbertrek visualization dependency and the pure-Python imbalance/OSDT
variant are not ported, and the pybind11 binding is replaced by the same
C ABI + PyO3 bridge used by the other vendored engines.
