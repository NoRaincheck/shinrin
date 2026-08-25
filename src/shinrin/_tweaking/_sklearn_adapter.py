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

from ._core import INF, Constraint, merge_constraints


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
        if tree.children_left[node] == -1:
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
    """All trees of a fitted forest; a bare tree is treated as size one.

    Also covers voting committees whose members emit class labels:
    ``RandomForestClassifier``, ``ExtraTreesClassifier``, ``BaggingClassifier``
    with tree bases, and ``AdaBoostClassifier`` (whose
    ``estimator_weights_`` are exposed via :attr:`weights`).
    """

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
        weights = getattr(estimator, "estimator_weights_", None)
        if weights is not None and len(weights) == len(self._trees):
            self.weights: np.ndarray | None = np.asarray(weights, dtype=float)
        else:
            self.weights = None

    @property
    def n_features(self) -> int:
        return int(self._trees[0].n_features_in_)

    def probability_leaf_values(
        self, class_index: int
    ) -> list[list[tuple[Constraint, float]]]:
        """Per-tree ``P(class | leaf)`` leaf paths for soft-vote tweaking."""
        return [_leaves_with_probability(t.tree_, class_index) for t in self._trees]

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((len(self._trees), X.shape[0]), dtype=int)
        for m, tree in enumerate(self._trees):
            raw = tree.predict(X)
            out[m] = [self._label_to_index[label] for label in raw]
        return out


def leaves_from_hist_predictor(predictor) -> list[tuple[Constraint, float]]:
    """``(constraint, leaf_value)`` paths of a HistGradientBoosting predictor.

    Leaf values are already shrunk by the learning rate. Categorical splits
    are not supported.
    """
    nodes = predictor.nodes
    leaves: list[tuple[Constraint, float]] = []
    stack: list[tuple[int, Constraint]] = [(0, {})]
    while stack:
        node_idx, path = stack.pop()
        node = nodes[node_idx]
        if bool(node["is_leaf"]):
            leaves.append((path, float(node["value"])))
            continue
        if bool(node["is_categorical"]):
            raise NotImplementedError(
                "Tweaking does not support categorical splits in "
                "HistGradientBoosting predictors"
            )
        feature = int(node["feature_idx"])
        threshold = float(node["num_threshold"])
        left = merge_constraints(path, {feature: (-INF, _largest_leq(threshold))})
        right = merge_constraints(path, {feature: (_smallest_gt(threshold), INF)})
        if left is not None:
            stack.append((int(node["left"]), left))
        if right is not None:
            stack.append((int(node["right"]), right))
    return leaves


def _gbm_base_score(estimator) -> float:
    """Initial log-odds of a binary GradientBoostingClassifier.

    Reproduces sklearn's deviance init (log-odds of the positive-class
    prior); verified against ``decision_function`` to machine precision.
    """
    prior = float(np.asarray(estimator.init_.class_prior_)[1])
    prior = min(max(prior, 1e-6), 1 - 1e-6)
    return float(np.log(prior / (1 - prior)))


class GradientBoostingView:
    """Stages of a fitted binary ``GradientBoostingClassifier``.

    Members emit continuous scores rather than votes:
    ``decision_function(x) = base_score + learning_rate * sum(tree outputs)``,
    verified empirically to machine precision. Each stage contributes one
    regression-tree output scaled by the learning rate.
    """

    def __init__(self, estimator):
        shape = estimator.estimators_.shape
        if len(shape) != 2 or shape[1] != 1:
            raise NotImplementedError(
                "Tweaking supports only binary GradientBoostingClassifier"
            )
        self._stages = [t for t in estimator.estimators_[:, 0]]
        self.scale = float(estimator.learning_rate)
        self.base_score = _gbm_base_score(estimator)
        self.leaf_values: list[list[tuple[Constraint, float]]] = [
            _leaves_with_values(t.tree_) for t in self._stages
        ]

    @property
    def n_features(self) -> int:
        return int(self._stages[0].n_features_in_)

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Per-stage contributions (learning-rate-scaled), shape ``(n_stages, n)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((len(self._stages), X.shape[0]), dtype=float)
        for m, tree in enumerate(self._stages):
            out[m] = self.scale * tree.predict(X).ravel()
        return out


def _leaves_with_values(tree) -> list[tuple[Constraint, float]]:
    """``(constraint, raw_leaf_value)`` paths of an sklearn regression tree."""
    leaves: list[tuple[Constraint, float]] = []
    stack: list[tuple[int, Constraint]] = [(0, {})]
    while stack:
        node, path = stack.pop()
        if tree.children_left[node] == -1:
            leaves.append((path, float(tree.value[node].ravel()[0])))
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


def _leaves_with_probability(tree, class_index: int) -> list[tuple[Constraint, float]]:
    """``(constraint, P(class_index | leaf))`` paths of an sklearn classifier tree."""
    leaves: list[tuple[Constraint, float]] = []
    stack: list[tuple[int, Constraint]] = [(0, {})]
    while stack:
        node, path = stack.pop()
        if tree.children_left[node] == -1:
            values = tree.value[node].ravel()
            proba = float(values[class_index] / values.sum())
            leaves.append((path, proba))
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


class HistGradientBoostingView:
    """Stages of a fitted binary ``HistGradientBoostingClassifier``.

    Leaf values already include the shrinkage factor; the model baseline is
    added on top. Verified against ``decision_function`` to machine
    precision for numeric features.
    """

    def __init__(self, estimator):
        self._estimator = estimator
        stages = estimator._predictors
        if any(len(stage) != 1 for stage in stages):
            raise NotImplementedError(
                "Tweaking supports only binary HistGradientBoostingClassifier"
            )
        self._stages = [stage[0] for stage in stages]
        self.scale = 1.0
        self.base_score = float(np.ravel(estimator._baseline_prediction)[0])
        self.leaves = None
        self.leaf_values: list[list[tuple[Constraint, float]]] = [
            leaves_from_hist_predictor(p) for p in self._stages
        ]

    @property
    def n_features(self) -> int:
        return int(self._estimator.n_features_in_)

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        known_cats = np.zeros((1, 8), dtype=np.uint32)
        fmap = np.arange(2, X.shape[1] + 2, dtype=np.uint32)
        out = np.empty((len(self._stages), X.shape[0]), dtype=float)
        for m, pred in enumerate(self._stages):
            out[m] = pred.predict(np.asfortranarray(X), known_cats, fmap, n_threads=1)
        return out
