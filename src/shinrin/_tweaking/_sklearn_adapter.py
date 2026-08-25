"""Adapter for scikit-learn decision trees and forests.

Leaf literal sets are derived from the compiled ``tree_`` arrays: going left
means ``x[f] <= threshold``, going right means ``x[f] > threshold``.

sklearn casts feature values to float32 before comparing them against the
float64 thresholds, so interval boundaries emitted here are adjusted to the
nearest float32 value that provably satisfies the strict/non-strict
comparison after that cast. Projections onto such boundaries therefore
re-route through the intended child under sklearn's own prediction code.
"""

from __future__ import annotations

import numpy as np

from ._core import Constraint, INF, merge_constraints


def _largest_leq(threshold: float) -> float:
    """Largest float32 value ``v`` with ``v <= threshold``."""
    v = np.float32(threshold)
    if float(v) > threshold:
        v = np.nextafter(v, np.float32(-np.inf))
    return float(v)


def _smallest_gt(threshold: float) -> float:
    """Smallest float32 value ``v`` with ``v > threshold``."""
    v = np.float32(threshold)
    while float(v) <= threshold:
        v = np.nextafter(v, np.float32(np.inf))
    return float(v)


def leaves_from_sklearn_tree(tree) -> list[tuple[Constraint, int]]:
    """``(constraint, class_index)`` leaf paths of a fitted sklearn tree."""
    leaves: list[tuple[Constraint, int]] = []
    stack: list[tuple[int, Constraint]] = [(0, {})]
    while stack:
        node, path = stack.pop()
        if tree.children_left[node] == -1:  # noqa: PLR2004 (leaf sentinel)
            label = int(np.argmax(tree.value[node]))
            leaves.append((path, label))
            continue
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        left = merge_constraints(path, {feature: (-INF, _largest_leq(threshold))})
        right = merge_constraints(path, {feature: (_smallest_gt(threshold), INF)})
        if left is not None:
            stack.append((int(tree.children_left[node]), left))
        if right is not None:
            stack.append((int(tree.children_right[node]), right))
    return leaves


class SklearnForestView:
    """All trees of a fitted forest; a bare tree is treated as size one."""

    def __init__(self, estimator):
        if hasattr(estimator, "estimators_"):
            self._trees = list(estimator.estimators_)
        else:
            self._trees = [estimator]
        self._classes = np.asarray(estimator.classes_)
        self._label_to_index = {c: i for i, c in enumerate(self._classes)}
        self.leaves: list[list[tuple[Constraint, int]]] = [
            leaves_from_sklearn_tree(t.tree_) for t in self._trees
        ]

    @property
    def n_features(self) -> int:
        return int(self._trees[0].n_features_in_)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((len(self._trees), X.shape[0]), dtype=int)
        for m, tree in enumerate(self._trees):
            raw = tree.predict(X)
            out[m] = [self._label_to_index[label] for label in raw]
        return out
