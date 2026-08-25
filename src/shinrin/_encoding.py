"""CatBoost-style target encoder for categorical features held as codes.

Categorical columns are encoded with smoothed target statistics
(``enc(c) = (sum_y(c) + smoothing * prior) / (count(c) + smoothing)``,
the CatBoost greedy/target-statistics formula) so that tree models can
consume them as plain numeric features.

Because an encoded category space is discrete, a trained split
``x_encoded <= t`` does *not* really compare against a continuous
threshold: it selects the *partition* of categories whose encoded values
fall at or below ``t``. Recovering that partition is what makes such
models interpretable again and what allows exporting them to ONNX
``ai.onnx.ml TreeEnsemble`` graphs using ``BRANCH_MEMBER`` nodes over raw
category codes (see :mod:`shinrin.categorical`). The encoder therefore
keeps its full statistics table and exposes the partition post-processing
(:meth:`TargetEncoder.partitions`, :meth:`TargetEncoder.members`) plus the
inverse mapping (:meth:`TargetEncoder.threshold_for_partition`).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class TargetEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features with smoothed target statistics.

    Parameters
    ----------
    categorical_features : list of int, optional
        Indices of the columns holding categorical features (as numeric
        category codes). ``None`` treats every column as categorical.
    smoothing : float, default=1.0
        Prior-blending weight ``m`` of the CatBoost-style statistic
        ``(sum_y(c) + m * prior) / (count(c) + m)``. Larger values shrink
        rare-category encodings towards the global prior.

    Attributes
    ----------
    categorical_features_ : list of int
        Sorted column indices treated as categorical.
    categories_ : dict[int, ndarray]
        Per categorical column: sorted observed category codes.
    encodings_ : dict[int, ndarray]
        Per categorical column: encoded target statistic aligned with
        ``categories_[column]``.
    prior_ : float
        Global (prior) target mean; also the encoding used for categories
        unseen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from shinrin._encoding import TargetEncoder
    >>> X = np.array([[0., 10.], [1., 20.], [0., 30.], [1., 40.]])
    >>> y = np.array([0., 1., 1., 1.])
    >>> enc = TargetEncoder(categorical_features=[0]).fit(X, y)
    >>> list(enc.categories_[0]), list(enc.encodings_[0])
    ([0.0, 1.0], [0.5, 1.0])
    """

    def __init__(self, categorical_features=None, smoothing=1.0):
        self.categorical_features = categorical_features
        self.smoothing = smoothing

    def fit(self, X, y):
        """Learn the per-category target statistics from ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix; categorical columns hold category codes.
        y : array-like of shape (n_samples,)
            Target. Non-numeric targets are factorized to integer codes
            before computing means.

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        if not np.issubdtype(y.dtype, np.number):
            _, y = np.unique(y, return_inverse=True)
        y = y.astype(np.float64)

        n_features = X.shape[1]
        feats = self.categorical_features
        if feats is None:
            feats = list(range(n_features))
        feats = sorted(int(f) for f in feats)
        if not feats or feats[0] < 0 or feats[-1] >= n_features:
            raise ValueError(
                f"categorical_features {feats} out of range for {n_features} columns"
            )
        smoothing = float(self.smoothing)
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")

        self.prior_ = float(y.mean())
        self.n_features_in_ = n_features
        self.categorical_features_ = feats
        self.categories_: dict[int, np.ndarray] = {}
        self.encodings_: dict[int, np.ndarray] = {}
        for f in feats:
            cats, inv = np.unique(X[:, f], return_inverse=True)
            sums = np.bincount(inv, weights=y, minlength=len(cats))
            counts = np.bincount(inv, minlength=len(cats)).astype(np.float64)
            self.categories_[f] = cats
            self.encodings_[f] = (sums + smoothing * self.prior_) / (counts + smoothing)
        return self

    def transform(self, X):
        """Replace categorical columns with their target-statistic values.

        Non-categorical columns pass through unchanged. Categories unseen
        during fit map to the prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix in the original (pre-encoding) convention.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Encoded feature matrix suitable for fitting tree models.
        """
        check_is_fitted(
            self, ["categorical_features_", "categories_", "encodings_", "prior_"]
        )
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self.n_features_in_}"
            )
        Xt = np.array(X, dtype=np.float64, copy=True)
        for f in self.categorical_features_:
            cats, encs = self.categories_[f], self.encodings_[f]
            col = Xt[:, f]
            loc = np.searchsorted(cats, col)
            loc_clipped = np.clip(loc, 0, len(cats) - 1)
            hit = cats[loc_clipped] == col
            col[:] = np.where(hit, encs[loc_clipped], self.prior_)
        return Xt

    # ------------------------------------------------------------------
    # Partition post-processing: encoded thresholds <-> category sets
    # ------------------------------------------------------------------

    def _tables(self, feature: int) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["categories_", "encodings_"])
        feature = int(feature)
        if feature not in self.categorical_features_:
            raise ValueError(
                f"feature {feature} is not categorical "
                f"(categorical features: {self.categorical_features_})"
            )
        return self.categories_[feature], self.encodings_[feature]

    def partitions(self, feature, threshold) -> np.ndarray:
        """Boolean mask over ``categories_[feature]`` selected by a split.

        Mirrors ``BRANCH_LEQ`` routing on the encoded column: category
        ``c`` takes the true branch iff ``encodings_[feature][c] <=
        threshold``.

        Parameters
        ----------
        feature : int
            Categorical column index.
        threshold : float
            Split threshold in the encoded space (e.g. taken from a
            fitted tree's ``tree_.threshold``).

        Returns
        -------
        ndarray of bool, shape ``(len(categories_[feature]),)``
            ``True`` where the category goes to the true/left branch.
        """
        encs = self._tables(feature)[1]
        return encs <= float(threshold)

    def members(self, feature, threshold) -> np.ndarray:
        """Category codes routed to the true branch by an encoded split.

        This is the recovered partition — the semantic meaning of the
        split in terms of original categories.

        Parameters
        ----------
        feature : int
            Categorical column index.
        threshold : float
            Split threshold in the encoded space.

        Returns
        -------
        ndarray
            Observed category codes ``c`` with ``encode(c) <= threshold``.
        """
        cats, _ = self._tables(feature)
        return cats[self.partitions(feature, threshold)]

    def threshold_for_partition(self, feature, member_codes) -> float:
        """Inverse of :meth:`members`: a threshold reproducing a partition.

        Finds an encoded-space threshold ``t`` such that
        ``{c : encode(c) <= t}`` equals ``member_codes``. An ``LEQ``
        threshold selects a *prefix* of the categories sorted by encoded
        value, so only such prefixes (exactly what splits trained on
        encoded columns induce) are representable; anything else raises.

        Parameters
        ----------
        feature : int
            Categorical column index.
        member_codes : array-like
            Category codes that should take the true branch.

        Returns
        -------
        float
            Equivalent encoded threshold.
        """
        cats, encs = self._tables(feature)
        wanted = np.isin(cats, np.asarray(member_codes))
        if not wanted.any():
            return float(np.nextafter(encs.min(), -np.inf))
        if wanted.all():
            return float(encs.max())
        order = np.argsort(encs, kind="stable")
        k = int(np.argmax(~wanted[order]))  # first non-member in sort order
        if wanted[order[k:]].any():
            raise ValueError(
                f"partition {member_codes!r} on feature {feature} is not "
                "expressible as a single encoded threshold: LEQ selects a "
                "prefix of the encoding-sorted categories"
            )
        boundary = encs[order[k]]
        if boundary <= encs[order[k - 1]]:
            raise ValueError(
                f"partition {member_codes!r} on feature {feature} is not "
                "expressible as a single encoded threshold: tied encodings "
                "straddle the partition boundary"
            )
        return float(encs[order[k - 1]])
