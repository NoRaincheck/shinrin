"""Pure-Python facade for the Mojo native tree extension.

This module mirrors the exact API surface of the Rust ``shinrin._native``
extension module (properties, keyword arguments and pickle support) on top
of the method-only Mojo core in ``shinrin._native_mojo_core``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shinrin import _native_mojo_core as _core

__all__ = [
    "DOUBLE",
    "DTYPE",
    "MSE",
    "BaseDenseSplitter",
    "ClassificationCriterion",
    "Criterion",
    "DepthFirstTreeBuilder",
    "MondrianSplitter",
    "PartialFitTreeBuilder",
    "Splitter",
    "Tree",
]

DTYPE = _core.DTYPE
DOUBLE = _core.DOUBLE


def _as_f32_2d(X: Any) -> np.ndarray:
    return np.ascontiguousarray(X, dtype=np.float32)


def _as_f64_2d(y: Any) -> np.ndarray:
    return np.ascontiguousarray(y, dtype=np.float64)


def _as_f64_1d(w: Any) -> np.ndarray:
    return np.ascontiguousarray(w, dtype=np.float64)


# =============================================================================
# Criterion
# =============================================================================


class Criterion:
    """Base class for split criteria (mirrors shinrin._native.Criterion)."""

    _core: Any

    def __reduce__(self):
        return (type(self), self._reduce_args(), {})

    def _reduce_args(self):
        raise NotImplementedError


class MSE(Criterion):
    def __init__(self, n_outputs: int, n_samples: int):
        self.n_outputs = int(n_outputs)
        self.n_samples = int(n_samples)
        self._core = _core.CoreCriterion(self.n_outputs, self.n_samples)

    def _reduce_args(self):
        return (self.n_outputs, self.n_samples)


class ClassificationCriterion(Criterion):
    def __init__(self, n_outputs: int, n_classes):
        self.n_outputs = int(n_outputs)
        classes = np.asarray(n_classes, dtype=np.intp)
        self._core = _core.CoreCriterion(self.n_outputs, classes)

    def _reduce_args(self):
        return (self.n_outputs, self.n_classes)

    @property
    def n_classes(self):
        return self._core.n_classes()


# =============================================================================
# Splitters
# =============================================================================


class Splitter:
    """Base splitter (mirrors shinrin._native.Splitter)."""

    def __init__(self, criterion, random_state):
        self._core = _core.CoreSplitter(_unwrap(criterion), random_state)
        self.criterion = criterion
        self.random_state = random_state

    def __reduce__(self):
        return (type(self), (self.criterion, self.random_state), {})


class BaseDenseSplitter(Splitter):
    pass


class MondrianSplitter(BaseDenseSplitter):
    pass


def _unwrap(obj):
    return obj._core if hasattr(obj, "_core") else obj


# =============================================================================
# Tree
# =============================================================================


class Tree:
    """Mondrian tree (mirrors shinrin._native.Tree)."""

    def __init__(self, n_features: int, n_classes, n_outputs: int):
        classes = np.asarray(n_classes, dtype=np.intp)
        self._core = _core.CoreTree(int(n_features), classes, int(n_outputs))

    # -- scalar properties ---------------------------------------------------

    @property
    def node_count(self):
        return self._core.node_count()

    @property
    def capacity(self):
        return self._core.capacity()

    @property
    def max_depth(self):
        return self._core.max_depth()

    @property
    def n_features(self):
        return self._core.n_features()

    @property
    def n_outputs(self):
        return self._core.n_outputs()

    @property
    def max_n_classes(self):
        return self._core.max_n_classes()

    @property
    def root(self):
        return self._core.root()

    @property
    def n_classes(self):
        return self._core.n_classes()

    # -- per-node arrays -----------------------------------------------------

    @property
    def children_left(self):
        return self._core.children_left()

    @property
    def children_right(self):
        return self._core.children_right()

    @property
    def feature(self):
        return self._core.feature()

    @property
    def threshold(self):
        return self._core.threshold()

    @property
    def impurity(self):
        return self._core.impurity()

    @property
    def n_node_samples(self):
        return self._core.n_node_samples()

    @property
    def weighted_n_node_samples(self):
        return self._core.weighted_n_node_samples()

    @property
    def tau(self):
        return self._core.tau()

    @property
    def lower_bounds(self):
        return self._core.lower_bounds()

    @property
    def upper_bounds(self):
        return self._core.upper_bounds()

    @property
    def variance(self):
        return self._core.variance()

    @property
    def mean(self):
        return self._core.mean()

    @property
    def base_value(self):
        return self._core.base_value()

    @property
    def value(self):
        return self._core.value()

    # -- inference ------------------------------------------------------------

    def apply(self, X):
        return self._core.apply(_as_f32_2d(X))

    def predict(self, X, return_std=False, is_regression=True):
        return self._core.predict(_as_f32_2d(X), bool(return_std), bool(is_regression))

    def decision_path(self, X):
        return self._core.decision_path(_as_f32_2d(X))

    def isolation_path_length(self, X):
        return self._core.isolation_path_length(_as_f32_2d(X))

    def weighted_decision_path(self, X):
        return self._core.weighted_decision_path(_as_f32_2d(X))

    def shap_values(self, X):
        return self._core.shap_values(_as_f32_2d(X))

    # -- construction helpers ---------------------------------------------------

    def populate_from_arrays(
        self,
        *,
        left_child,
        right_child,
        feature,
        threshold,
        n_node_samples,
        value,
        tau,
        lower_bounds,
        upper_bounds,
    ):
        self._core.populate_from_arrays(
            (
                np.ascontiguousarray(left_child, dtype=np.float64),
                np.ascontiguousarray(right_child, dtype=np.float64),
                np.ascontiguousarray(feature, dtype=np.float64),
                np.ascontiguousarray(threshold, dtype=np.float64),
                np.ascontiguousarray(n_node_samples, dtype=np.float64),
                np.ascontiguousarray(value, dtype=np.float64),
                np.ascontiguousarray(tau, dtype=np.float32),
                np.ascontiguousarray(lower_bounds, dtype=np.float32),
                np.ascontiguousarray(upper_bounds, dtype=np.float32),
            )
        )

    # -- pickle support ---------------------------------------------------------

    def __getstate__(self):
        return self._core.getstate()

    def __setstate__(self, state):
        self._core.setstate(state)

    def __reduce__(self):
        args = (
            self._core.n_features(),
            self._core.n_classes(),
            self._core.n_outputs(),
        )
        return (Tree, args, self._core.getstate())


# =============================================================================
# Builders
# =============================================================================


class DepthFirstTreeBuilder:
    def __init__(self, splitter, min_samples_split, max_depth):
        self._core = _core.CoreDepthFirstBuilder(
            _unwrap(splitter), int(min_samples_split), int(max_depth)
        )
        self.splitter = splitter

    def build(
        self,
        tree,
        X,
        y,
        sample_weight=None,
        X_idx_sorted=None,
    ):
        X = _as_f32_2d(X)
        y = _as_f64_2d(y)
        sw = None if sample_weight is None else _as_f64_1d(sample_weight)
        self._core.build(tree._core, X, y, sw, X_idx_sorted)


class PartialFitTreeBuilder:
    def __init__(self, min_samples_split, max_depth, random_state):
        self._core = _core.CorePartialFitBuilder(
            int(min_samples_split), int(max_depth), random_state
        )

    def build(
        self,
        tree,
        X,
        y,
        sample_weight=None,
        X_idx_sorted=None,
    ):
        _ = sample_weight
        _ = X_idx_sorted
        X = _as_f32_2d(X)
        y = _as_f64_2d(y)
        self._core.build(tree._core, X, y)
