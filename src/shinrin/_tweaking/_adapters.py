"""Adapters translating fitted tree models into core-solver leaf inventories.

Every adapter produces a *model view*: the per-model list of
``(constraint, label)`` root-to-leaf literal sets plus batch prediction over
all models for verification. Labels are integer-encoded exactly as the
underlying trees emit them; mapping to original class labels happens in the
facade via ``classes_``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ._core import Constraint, merge_constraints


def leaves_from_binary_tree(node: dict[str, Any]) -> list[tuple[Constraint, int]]:
    """Extract ``(constraint, label)`` leaf paths from a binary-tree dict.

    Handles both the SPOTSET (treeFARMS) and SPOT (GOSDT) nested-dict formats:
    internal nodes carry ``feature`` / ``true`` / ``false`` where the ``true``
    branch means ``x[feature] == 1``; leaves carry ``prediction``.
    """
    leaves: list[tuple[Constraint, int]] = []
    stack: list[tuple[dict[str, Any], Constraint]] = [(node, {})]
    while stack:
        current, path = stack.pop()
        if "prediction" in current:
            leaves.append((path, int(current["prediction"])))
            continue
        feature = int(current["feature"])
        merged_true = merge_constraints(path, {feature: (1.0, 1.0)})
        merged_false = merge_constraints(path, {feature: (0.0, 0.0)})
        if merged_true is not None:
            stack.append((current["true"], merged_true))
        if merged_false is not None:
            stack.append((current["false"], merged_false))
    return leaves


def classify_binary_tree(node: dict[str, Any], x: np.ndarray) -> int:
    """Traverse a binary-tree dict for one sample."""
    while "prediction" not in node:
        node = node["true"] if x[int(node["feature"])] == 1 else node["false"]
    return int(node["prediction"])


def _n_features_of(estimator) -> int:
    if hasattr(estimator, "n_features_in_"):
        return int(estimator.n_features_in_)
    return int(estimator.n_features_)


class SpotsetView:
    """All trees of a fitted :class:`shinrin.SPOTSETClassifier`."""

    def __init__(self, clf):
        self._trees = [clf[i] for i in range(clf.n_trees_)]
        self._n_features = _n_features_of(clf)
        self.leaves: list[list[tuple[Constraint, int]]] = [
            leaves_from_binary_tree(t.source) for t in self._trees
        ]

    @property
    def n_features(self) -> int:
        return self._n_features

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([t.predict(X).astype(int) for t in self._trees])


class SpotView:
    """The optimal tree of a fitted :class:`shinrin.SPOTClassifier`."""

    def __init__(self, clf):
        self._model = json.loads(clf.get_result()["models_string"])[0]
        self._n_features = _n_features_of(clf)
        self.leaves: list[list[tuple[Constraint, int]]] = [
            leaves_from_binary_tree(self._model)
        ]

    @property
    def n_features(self) -> int:
        return self._n_features

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array(
            [[classify_binary_tree(self._model, row) for row in X]], dtype=int
        )
