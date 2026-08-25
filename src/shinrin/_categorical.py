"""Automatic categorical-feature awareness shared by shinrin estimators.

Two building blocks live here:

- :func:`resolve_categorical_mask` — the heuristic detector. A column is
  treated as categorical when every value is integral and the number of
  unique values satisfies ``2 <= n_unique <= max_categories`` (default 32).
- :class:`TargetStatisticsEncoder` — a CatBoost-style smoothed
  target-statistic encoder. Each detected categorical column is replaced by
  a single numeric column holding the regularized target statistic of its
  category::

      stat(c) = (sum_y(c) + smoothing * global_mean) / (count(c) + smoothing)

  For continuous targets this is the smoothed per-category mean; for
  classification targets the class labels are their integer indices, so the
  statistic is the class-index-weighted mean ``E[class | category]``
  (binary targets reduce to the familiar ``P(positive | category)``).
  Categories unseen at fit time map to the global prior.

Encoding one numeric column per categorical feature keeps the feature
count unchanged, so downstream consumers (SHAP shapes, ONNX export,
anomaly scores) operate on the same axis layout as before. Splits on the
encoded axis correspond to groupings of categories by target rate — the
same partition family that LightGBM's sorted-category scan enumerates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TargetStatisticsEncoder", "resolve_categorical_mask"]

_AUTO = "auto"


def resolve_categorical_mask(
    X: np.ndarray,
    spec: str | bool | None | np.ndarray,
    max_categories: int = 32,
) -> np.ndarray | None:
    """Resolve which columns of ``X`` should be treated as categorical.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training (or prediction) features as a 2-D numeric array.
    spec : {"auto", True, False, None} or array-like of int/bool
        - ``"auto"`` (or ``True``): apply the detection heuristic.
        - ``False`` / ``None``: no categorical handling; returns ``None``.
        - array-like: explicit selection — either a boolean mask of length
          ``n_features`` or an array of categorical column indices.
    max_categories : int, default=32
        Upper bound on the number of unique integer values a column may
        have to still qualify as categorical under ``"auto"``.

    Returns
    -------
    ndarray of bool or None
        Boolean mask of length ``n_features``, or ``None`` when categorical
        handling is disabled.
    """
    n_features = X.shape[1]

    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        if spec != _AUTO:
            raise ValueError(
                f"categorical_features must be 'auto', None, False, a boolean "
                f"mask or integer indices; got {spec!r}"
            )
        mask = np.zeros(n_features, dtype=bool)
        for j in range(n_features):
            if _looks_categorical(X[:, j], max_categories):
                mask[j] = True
        return mask
    if spec is True:
        return resolve_categorical_mask(X, _AUTO, max_categories)

    # Explicit user specification: boolean mask or integer index array.
    explicit = np.asarray(spec)
    if explicit.dtype == bool:
        if explicit.shape != (n_features,):
            raise ValueError(
                f"categorical_features boolean mask has shape {explicit.shape}, "
                f"expected ({n_features},)"
            )
        return explicit.copy()
    if explicit.ndim != 1:
        raise ValueError(
            "categorical_features must be a 1-D array of column indices; "
            f"got shape {explicit.shape}"
        )
    indices = np.asarray(explicit, dtype=np.intp).ravel()
    if indices.size:
        ordered = np.sort(indices)
        if int(ordered[0]) < 0 or int(ordered[-1]) >= n_features:
            raise ValueError(
                f"categorical_features contains an out-of-bounds column index "
                f"for {n_features} features"
            )
    mask = np.zeros(n_features, dtype=bool)
    mask[indices] = True
    return mask


def _looks_categorical(column: np.ndarray, max_categories: int) -> bool:
    """Heuristic: purely integral values with 2..max_categories uniques."""
    values = np.asarray(column).ravel()
    finite = values[np.isfinite(values)]
    if finite.size != values.size or finite.size == 0:
        # NaN/inf anywhere disqualifies the column.
        return False
    if not np.all(finite == np.round(finite)):
        return False
    n_unique = np.unique(finite).size
    return 2 <= n_unique <= max_categories


class TargetStatisticsEncoder:
    """CatBoost-style smoothed target-statistic encoder for categorical columns.

    Fits one mapping per selected column from raw category value to a
    regularized target statistic (see the module docstring for the formula).
    :meth:`transform` replaces each selected column in-place with its
    encoded values, preserving the matrix shape and returning a
    C-contiguous ``float32`` array ready for the native tree builders.

    Parameters
    ----------
    smoothing : float, default=1.0
        Prior strength ``s`` of the m-estimate shrinkage toward the global
        mean. Larger values pull rare categories harder toward the prior.
    """

    def __init__(self, smoothing: float = 1.0):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> TargetStatisticsEncoder:
        """Learn per-column category statistics from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features (raw, unencoded).
        y : ndarray of shape (n_samples,)
            Targets. Continuous values are used as-is; classification
            targets are expected to already be integer-encoded (the caller
            label-factorizes before fitting).
        mask : ndarray of shape (n_features,) of bool
            Columns to treat as categorical.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )

        self.prior_ = float(y.mean()) if y.size else 0.0
        smoothing = self.smoothing
        self.encodings_: list[tuple[np.ndarray, np.ndarray] | None] = []
        self.mask_ = np.asarray(mask, dtype=bool).copy()

        for j in range(X.shape[1]):
            if not self.mask_[j]:
                self.encodings_.append(None)
                continue
            column = X[:, j].astype(np.float64, copy=False)
            keys, inverse, counts = np.unique(
                column, return_inverse=True, return_counts=True
            )
            sums = np.bincount(inverse, weights=y, minlength=keys.size)
            stats = (sums + smoothing * self.prior_) / (counts + smoothing)
            order = np.argsort(keys)
            self.encodings_.append((keys[order], stats[order]))

        self.n_features_in_ = X.shape[1]
        return self

    @property
    def active_(self) -> bool:
        """Whether any column is actually being encoded."""
        return bool(getattr(self, "mask_", np.zeros(0, dtype=bool)).any())

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace categorical columns with their fitted target statistics.

        Values never seen during fit map to the global prior. Returns a
        C-contiguous ``float32`` array of the same shape as the input.
        """
        if not hasattr(self, "encodings_"):
            raise RuntimeError("TargetStatisticsEncoder has not been fitted")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected 2-D input with {self.n_features_in_} features, "
                f"got shape {X.shape}"
            )
        # Always copy: callers may pass their (check_X_y-validated, often
        # zero-copy) training buffer, which must never be mutated in place.
        out = np.array(X, dtype=np.float32, order="C", copy=True)
        for j, encoding in enumerate(self.encodings_):
            if encoding is None:
                continue
            keys, stats = encoding
            column = out[:, j].astype(np.float64, copy=False)
            loc = np.searchsorted(keys, column)
            loc_clipped = np.clip(loc, 0, keys.size - 1)
            matched = keys[loc_clipped] == column
            out[:, j] = np.where(matched, stats[loc_clipped], self.prior_)
        return out
