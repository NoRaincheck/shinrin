"""NumPy preprocessing transforms for the vendored TabM model.

Adapted from chapman73/tabm-lightning (see NOTICE) and reimplemented
without PyTorch. All transforms are fitted once during ``fit`` and are
not trained further.
"""

from __future__ import annotations

import numpy as np


def detect_categorical_features(
    X: np.ndarray,
    cardinality_threshold: int = 32,
    categorical_indices: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Detect which features are categorical based on cardinality.

    A feature is categorical when it is listed in ``categorical_indices``
    or has at most ``cardinality_threshold`` unique values.

    Returns:
        ``(categorical_indices, numerical_indices, cardinalities)``.
    """
    explicit = set(categorical_indices or ())
    categorical: list[int] = []
    numerical: list[int] = []
    cardinalities: list[int] = []
    for col in range(X.shape[1]):
        values = X[:, col]
        values = values[~np.isnan(values)]
        n_unique = len(np.unique(values))
        if col in explicit or n_unique <= cardinality_threshold:
            categorical.append(col)
            cardinalities.append(n_unique)
        else:
            numerical.append(col)
    return categorical, numerical, cardinalities


class QuantileTransform:
    """Map features to their quantile position in ``[0, 1]``."""

    def __init__(self, num_quantiles: int = 100) -> None:
        self.num_quantiles = num_quantiles
        self.boundaries_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> QuantileTransform:
        probs = np.linspace(0.0, 1.0, self.num_quantiles + 1)[1:-1]
        bounds = []
        for col in range(X.shape[1]):
            values = X[:, col]
            values = values[~np.isnan(values)]
            if len(values):
                bounds.append(np.quantile(values, probs))
            else:
                bounds.append(np.zeros(len(probs)))
        self.boundaries_ = np.asarray(bounds, dtype=np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.boundaries_ is not None
        out = np.empty_like(X, dtype=np.float32)
        for col in range(X.shape[1]):
            idx = np.searchsorted(self.boundaries_[col], X[:, col], side="right")
            out[:, col] = idx.astype(np.float32) / self.num_quantiles
        return out


class AsinhTransform:
    """Apply ``asinh``, compressing heavy tails while preserving sign."""

    def fit(self, X: np.ndarray) -> AsinhTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asinh(X).astype(np.float32)


class StandardScalerTransform:
    """Subtract the mean and divide by the standard deviation."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScalerTransform:
        self.mean_ = np.nanmean(X, axis=0).astype(np.float32)
        std = np.nanstd(X, axis=0).astype(np.float32)
        self.std_ = np.where(std == 0, np.float32(1.0), std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return ((X - self.mean_) / self.std_).astype(np.float32)


def build_num_bins(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Compute quantile bin edges per feature (adapted from tabm-lightning).

    Ensures every feature has at least two unique, increasing edges.
    """
    edges_list: list[np.ndarray] = []
    probs = np.linspace(0.0, 1.0, n_bins + 1)
    for col in range(X.shape[1]):
        values = X[:, col]
        values = values[~np.isnan(values)]
        if len(values):
            edges = np.unique(np.quantile(values, probs))
        else:
            edges = np.array([0.0])
        if len(edges) < 2:
            center = float(edges[0]) if len(edges) else 0.0
            edges = np.array([center - 0.1, center + 0.1])
        elif len(edges) < n_bins + 1:
            edges = np.linspace(edges[0], edges[-1], n_bins + 1)
        edges_list.append(edges.astype(np.float64))
    return edges_list


class PiecewiseLinearEncoder:
    """Non-trainable piecewise-linear encoding (compact layout).

    For a feature with ``M`` bins and edges ``e_0..e_M`` the encoding is::

        comp_j(x) = clip((x - e_j) / (e_{j+1} - e_j))

    where ``comp_0`` is clamped above at 1, interior components are clamped
    to ``[0, 1]`` and the last component is clamped below at 0. Features
    with a single bin use unclamped min-max scaling. Matches the semantics
    of ``rtdl_num_embeddings.PiecewiseLinearEncoding`` (Apache-2.0).
    """

    def __init__(self, bins: list[np.ndarray]) -> None:
        self.bins_ = bins
        self.offsets_ = np.cumsum([0] + [len(b) - 1 for b in bins])

    @property
    def width(self) -> int:
        return int(self.offsets_[-1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.empty((n, self.width), dtype=np.float32)
        for f, edges in enumerate(self.bins_):
            m = len(edges) - 1
            lo, hi = self.offsets_[f], self.offsets_[f + 1]
            block = out[:, lo:hi]
            if m == 1:
                block[:, 0] = (X[:, f] - edges[0]) / (edges[1] - edges[0])
                continue
            widths = np.diff(edges)
            t = (X[:, f, None] - edges[None, :-1]) / widths[None, :]
            block[:, 0] = np.minimum(t[:, 0], 1.0)
            if m > 2:
                block[:, 1 : m - 1] = np.clip(t[:, 1 : m - 1], 0.0, 1.0)
            block[:, m - 1] = np.maximum(t[:, m - 1], 0.0)
        return out
