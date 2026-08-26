"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from __future__ import division


import numbers
from abc import ABC
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from shinrin._compat.sklearn_base import BaseEstimator
from shinrin._compat.sklearn_base import ClassifierMixin
from shinrin._compat.sklearn_base import RegressorMixin
from shinrin._compat.sklearn_preprocessing import LabelEncoder
from shinrin._compat.sklearn_utils import check_random_state
from shinrin._compat.sklearn_utils_class_weight import compute_sample_weight
from shinrin._compat.sklearn_utils_multiclass import check_classification_targets
from shinrin._compat.sklearn_utils_validation import check_array
from shinrin._compat.sklearn_utils_validation import check_is_fitted
from shinrin._compat.sklearn_utils_validation import check_X_y
from shinrin._compat.sklearn_exceptions import NotFittedError
from shinrin._categorical import TargetStatisticsEncoder, resolve_categorical_mask

from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import DepthFirstTreeBuilder
from ._tree import PartialFitTreeBuilder
from ._tree import Tree
from . import _tree, _splitter, _criterion

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"classification": _criterion.ClassificationCriterion}
CRITERIA_REG = {"mse": _criterion.MSE}

SPLITTERS = {"mondrian": _splitter.MondrianSplitter}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(ABC, BaseEstimator):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 random_state,
                 class_weight=None,
                 path_smoothing=False,
                 categorical_features=None,
                 max_categories=32):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.class_weight = class_weight
        self.path_smoothing = path_smoothing
        self.categorical_features = categorical_features
        self.max_categories = max_categories

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        random_state = check_random_state(self.random_state)
        if check_input:
            X, y = check_X_y(X, y, dtype=DTYPE, multi_output=False)

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=np.int64)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Automatically detect + target-encode integer-coded categorical
        # columns before growing the tree (no-op unless anything is found).
        encoder = self._fit_categorical_encoder(X, y[:, 0])
        self._cat_encoder_ = encoder if encoder is not None and encoder.active_ else None
        if self._cat_encoder_ is not None:
            X = self._cat_encoder_.transform(X)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
                                                         n_samples)
        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                random_state)
        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)
        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        max_depth)
        builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _fit_categorical_encoder(self, X, y_stats):
        """Resolve the categorical mask and fit the target-statistic encoder.

        Returns ``None`` when categorical handling is disabled, no column
        qualifies, or the tree was converted from another model (in which
        case splits live in the original feature space and must not be
        reinterpreted).
        """
        if getattr(self, "_onnx_converted", False):
            self.categorical_features_ = None
            return None
        mask = resolve_categorical_mask(
            X, self.categorical_features, self.max_categories)
        self.categorical_features_ = mask
        if mask is None or not mask.any():
            return None
        return TargetStatisticsEncoder().fit(X, y_stats, mask)

    def transform_categoricals(self, X):
        """Encode integer-coded categorical columns of ``X`` for this model.

        Applies the fitted CatBoost-style target-statistic mapping to the
        columns selected by ``categorical_features`` during ``fit``. Columns
        without categorical handling pass through unchanged. Values unseen
        at fit time map to the global prior.

        Useful when a downstream consumer (e.g. ONNX export) operates in
        the model's internal encoded space.
        """
        encoder = getattr(self, "_cat_encoder_", None)
        if encoder is None:
            return np.asarray(X, dtype=np.float32)
        return encoder.transform(X)

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.tree_ is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        encoder = getattr(self, "_cat_encoder_", None)
        if encoder is not None:
            X = encoder.transform(X)

        return X

    def _resolve_path_smoothing(self, path_smoothing):
        """Resolve the effective prediction mode for a predict-time call."""
        if path_smoothing is None:
            return bool(getattr(self, "path_smoothing", False))
        return bool(path_smoothing)

    def predict(self, X, check_input=True, return_std=False,
                return_anomaly=False, return_shap=False, path_smoothing=None):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_std : boolean, (default=False)
            Whether or not to return the standard deviation.

        return_anomaly : boolean, (default=False)
            If True, return the Isolation Forest anomaly score for each sample.
            The anomaly score is computed as the average path length across
            all trees in the forest, normalized by the average path length of
            random trees built on the same dataset.

        return_shap : boolean, (default=False)
            If True, return TreeSHAP values for each sample.
            The SHAP values explain the contribution of each feature to the
            prediction relative to the base value (root prediction).

        path_smoothing : boolean, optional
            Override the estimator's ``path_smoothing`` setting for this
            call. ``None`` (default) uses the value chosen at construction
            time. See the class docstring for what the mode means.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        anomaly_scores : array of shape = [n_samples], optional
            Returned if ``return_anomaly=True``. Higher values indicate more
            anomalous samples.
        shap_values : array of shape = [n_samples, n_features], optional
            Returned if ``return_shap=True``. Each value represents the
            contribution of the corresponding feature to the prediction.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        smoothing = self._resolve_path_smoothing(path_smoothing)

        # Classification
        if isinstance(self, ClassifierMixin):
            if return_std:
                raise ValueError(
                    "return_std is not supported for classifiers. "
                    "Use MondrianTreeRegressor for standard deviation estimates."
                )
            prediction = self.classes_[self.predict_proba(
                X, path_smoothing=smoothing).argmax(axis=1)]
        # Regression
        else:
            mean_and_std = self.tree_.predict(
                X, return_std=return_std, is_regression=True,
                path_smoothing=smoothing)
            prediction = mean_and_std[0]

        # Build return tuple consistently
        parts = [prediction]
        if return_std:
            parts.append(mean_and_std[1])
        if return_anomaly:
            parts.append(self._compute_anomaly(X))
        if return_shap:
            parts.append(self._compute_shap(X, path_smoothing=smoothing))

        # Filter out None values
        parts = [p for p in parts if p is not None]

        if len(parts) == 1:
            return parts[0]
        return tuple(parts)

    def _compute_anomaly(self, X):
        """Compute per-sample Isolation Forest anomaly scores for a single tree.

        Returns the raw average path length for each sample. The forest-level
        normalization (dividing by c(n) and applying 2^(-x/c(n))) is done at
        the forest level in ``pred_anomaly``.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input=True)
        return self.tree_.isolation_path_length(X)

    def _compute_shap(self, X, check_input=True, path_smoothing=None):
        """Compute TreeSHAP values for each sample.

        Decomposes this tree's prediction (under the effective
        ``path_smoothing`` mode) into feature contributions such that:
            prediction = base_value + sum(shap_values)

        For regression, returns shape (n_samples, n_features).
        For classification, returns shape (n_samples, n_features, n_classes)
        with per-class SHAP values that satisfy:
            predict_proba[i, c] = base_value[c] + sum(shap_values[i, :, c])

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input samples.

        path_smoothing : boolean, optional
            Override the estimator's ``path_smoothing`` setting for this
            call. ``None`` (default) uses the value chosen at construction
            time.

        Returns
        -------
        shap_values : ndarray
            SHAP values. Shape is (n_samples, n_features) for regression
            or (n_samples, n_features, n_classes) for classification.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        is_regression = not isinstance(self, ClassifierMixin)
        smoothing = self._resolve_path_smoothing(path_smoothing)

        if not smoothing:
            return self._hard_path_shap(
                X, is_regression=is_regression,
                n_samples=n_samples, n_features=n_features)

        # Get weighted decision path: (n_samples, n_nodes) with weights
        weighted_path = self.weighted_decision_path(X)
        if hasattr(weighted_path, 'toarray'):
            weighted_path = weighted_path.toarray()

        if is_regression:
            # Regression: node values are means
            node_values = self.tree_.mean[:self.tree_.node_count]
            base_val = node_values[0]
            shap = np.zeros((n_samples, n_features))

            for i in range(n_samples):
                path_mask = weighted_path[i] > 0
                path_node_indices = np.where(path_mask)[0]
                if len(path_node_indices) == 0:
                    continue

                path_features = self.tree_.feature[path_node_indices]
                path_weights = weighted_path[i, path_node_indices]

                feat_contribs = np.zeros(n_features)
                leaf_contrib = 0.0

                for node_idx, feat, weight in zip(
                    path_node_indices, path_features, path_weights):
                    node_contrib = weight * (node_values[node_idx] - base_val)
                    if feat < 0 or feat >= n_features:  # Leaf node
                        leaf_contrib += node_contrib
                    else:
                        feat_contribs[feat] += node_contrib

                if leaf_contrib != 0.0:
                    total_feat_weight = sum(
                        w for f, w in zip(path_features, path_weights)
                        if 0 <= f < n_features)
                    if total_feat_weight > 0:
                        for feat, weight in zip(path_features, path_weights):
                            if 0 <= feat < n_features:
                                shap[i, feat] += leaf_contrib * (weight / total_feat_weight)
                    else:
                        shap[i] += leaf_contrib / n_features

                shap[i] += feat_contribs
        else:
            # Classification: compute per-class SHAP values
            # Use class-conditional TreeSHAP (Lundberg et al. 2020)
            n_classes = self.n_classes_
            shap = np.zeros((n_samples, n_features, n_classes))

            for c in range(n_classes):
                # Build binary value array: P(class=c | node)
                n_node_samples_arr = self.tree_.n_node_samples[:self.tree_.node_count]
                values = self.tree_.value[:self.tree_.node_count, :, :]
                class_values = np.zeros(self.tree_.node_count)
                for j in range(self.tree_.node_count):
                    total = n_node_samples_arr[j]
                    if total > 0:
                        class_values[j] = values[j, 0, c] / total
                    else:
                        class_values[j] = 0.0

                base_val_c = class_values[0]

                for i in range(n_samples):
                    path_mask = weighted_path[i] > 0
                    path_node_indices = np.where(path_mask)[0]
                    if len(path_node_indices) == 0:
                        continue

                    path_features = self.tree_.feature[path_node_indices]
                    path_weights = weighted_path[i, path_node_indices]

                    feat_contribs = np.zeros(n_features)
                    leaf_contrib = 0.0

                    for node_idx, feat, weight in zip(
                        path_node_indices, path_features, path_weights):
                        node_contrib = weight * (class_values[node_idx] - base_val_c)
                        if feat < 0 or feat >= n_features:  # Leaf node
                            leaf_contrib += node_contrib
                        else:
                            feat_contribs[feat] += node_contrib

                    if leaf_contrib != 0.0:
                        total_feat_weight = sum(
                            w for f, w in zip(path_features, path_weights)
                            if 0 <= f < n_features)
                        if total_feat_weight > 0:
                            for feat, weight in zip(path_features, path_weights):
                                if 0 <= feat < n_features:
                                    shap[i, feat, c] += leaf_contrib * (weight / total_feat_weight)
                        else:
                            shap[i, :, c] += leaf_contrib / n_features

                    shap[i, :, c] += feat_contribs

            return shap

        return shap

    def _hard_path_shap(self, X, is_regression, n_samples, n_features):
        """TreeSHAP decomposition of the constant (leaf) prediction.

        Each split along the routed root-to-leaf path is attributed with
        the change in predicted value it introduces, so that
        ``prediction = root value + sum(shap_values)`` holds exactly.
        """
        n_classes = self.n_classes_
        tree = self.tree_

        def _val(nid):
            if is_regression:
                return tree.mean[nid]
            total = tree.n_node_samples[nid]
            if total <= 0:
                return np.zeros(n_classes)
            return tree.value[nid, 0, :n_classes] / total

        paths = self.decision_path(X, check_input=False)
        if hasattr(paths, 'toarray'):
            paths = paths.toarray()

        shap = np.zeros(
            (n_samples, n_features, n_classes)
            if not is_regression else (n_samples, n_features))

        for i in range(n_samples):
            path_node_indices = np.where(paths[i] > 0)[0]
            # Walk consecutive pairs; each internal node contributes the
            # value change between itself and the child it routes to.
            for parent, child in zip(path_node_indices[:-1],
                                     path_node_indices[1:]):
                feat = tree.feature[parent]
                if feat < 0 or feat >= n_features:
                    continue
                contrib = _val(child) - _val(parent)
                if is_regression:
                    shap[i, feat] += contrib
                else:
                    shap[i, feat, :] += contrib

        return shap

    def pred_contribs(self, X, check_input=True, path_smoothing=None):
        """Return TreeSHAP values including the base value.

        This method returns SHAP values such that the sum of SHAP values
        plus the base value equals the model prediction:
            prediction = base_value + sum(shap_values)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        path_smoothing : boolean, optional
            Override the estimator's ``path_smoothing`` setting for this
            call. ``None`` (default) uses the value chosen at construction
            time.

        Returns
        -------
        shap_values : array
            The last column/axis contains the base value (root prediction).
            For regression, shape is (n_samples, n_features + 1).
            For classification with K classes, shape is (n_samples, n_features + 1, K).
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        shap = self._compute_shap(X, path_smoothing=path_smoothing)

        if isinstance(self, ClassifierMixin):
            # shap is (n_samples, n_features, n_classes)
            n_samples = X.shape[0]
            n_classes = shap.shape[-1]
            base = self.tree_.base_value[:n_classes]
            # Stack shap values with base value along feature axis
            result = np.zeros((n_samples, shap.shape[1] + 1, n_classes))
            result[:, :-1, :] = shap
            result[:, -1, :] = base[np.newaxis, :]
            return result
        else:
            base = self.tree_.base_value
            # Broadcast base value to match number of samples
            base_expanded = np.broadcast_to(base, (X.shape[0],) + base.shape)
            return np.concatenate([shap, base_expanded], axis=1)

    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)


class BaseMondrianTree(BaseDecisionTree):
    """A Mondrian tree.

    The splits in a mondrian tree regressor differ from the standard regression
    tree in the following ways.

    At fit time:
        - Splits are done independently of the labels.
        - The candidate feature is drawn with a probability proportional to the
          feature range.
        - The candidate threshold is drawn from a uniform distribution
          with the bounds equal to the bounds of the candidate feature.
        - The time of split is also stored which is proportional to the
          inverse of the size of the bounding-box.

    At prediction time:
        - **Constant (default).** Each sample is routed down a single
          root-to-leaf path by hard threshold comparisons and receives the
          leaf's stored value (leaf mean for regression, normalized class
          counts for classification). This matches the piecewise-constant
          behaviour of scikit-learn's tree and forest predictors, and is
          exactly what the plain ONNX ``ai.onnx.ml`` tree-ensemble export
          computes. It is an *opinionated default*: it deliberately deviates
          from the "pure" Mondrian-process prediction described below.
        - **Path smoothing** (`path_smoothing=True`). Every node on the path
          from root to leaf is given a weight while making predictions.
          At each node, the probability of an unseen sample splitting from
          that node is calculated. The farther the sample is away from the
          bounding box, the more probable that it will split away.
          For every node, the probability that an unseen sample has not
          split before reaching that node and the probability that it will
          split away at that particular node are multiplied to give a weight.
          This is the predictor of the original Mondrian-forest process;
          enable it when you want the statistically pure estimator or its
          uncertainty estimates.

    Parameters
    ----------
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    path_smoothing : bool, optional (default=False)
        Prediction mode. With the default ``False``, predictions are
        piecewise-constant leaf values (scikit-learn-consistent; see above).
        With ``True``, predictions use the pure Mondrian-process weighting
        over every node on the decision path. This only affects
        ``predict`` / ``predict_proba``; SHAP contributions and anomaly
        scores always use the Mondrian node weights regardless of this
        setting.

    categorical_features : "auto", None, bool or array-like, optional (default="auto")
        Automatic categorical-feature awareness. Columns whose values are
        purely integral with at most ``max_categories`` unique values are
        treated as categorical and replaced by CatBoost-style smoothed
        target statistics before growing the tree. This makes split-point
        geometry depend on target structure rather than arbitrary integer
        codes.

        - ``"auto"``: apply the detection heuristic (default).
        - boolean mask / integer indices: explicit column selection.
        - ``None`` / ``False``: disable; every column is treated numerically.

    max_categories : int, optional (default=32)
        Cardinality cap used by the ``"auto"`` heuristic.
    """
    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Trees.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianTree
        """
        random_state = check_random_state(self.random_state)
        X, y = check_X_y(X, y, dtype=DTYPE, multi_output=False, order="C")
        is_classifier = isinstance(self, ClassifierMixin)
        random_state = check_random_state(self.random_state)
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)

        # This is necessary to rebuild the tree if partial_fit is called
        # after fit.
        first_call = not hasattr(self, "first_")
        if not hasattr(self, "first_"):
            self.first_ = True

        # Skip rebuilding if tree was converted from ONNX/sklearn
        # Only skip if _onnx_converted is True (set by from_model)
        if getattr(self, '_onnx_converted', False):
            if hasattr(self, 'tree_') and hasattr(self.tree_, 'n_node_samples'):
                if self.tree_.n_node_samples.sum() > 0:
                    # Tree was already built (converted), skip rebuilding
                    return self

        if is_classifier:
            check_classification_targets(y)

            # First call to partial_fit
            if first_call:
                if len(y) == 1 and classes is None:
                    raise ValueError("Unable to infer classes. Should be "
                                     "provided at the first call to partial_fit.")
                self.le_ = LabelEncoder()
                if classes is not None:
                    self.le_.fit(classes)
                else:
                    self.le_.fit(y)
                self.classes_ = self.le_.classes_
            y = self.le_.transform(y)
            n_classes = [len(self.le_.classes_)]
        else:
            n_classes = [1]

        # To be consistent with sklearns tree architecture, we reshape.
        y = np.array(y, dtype=np.float64)
        y = np.reshape(y, (-1, 1))

        # First call to partial_fit, initalize tree
        if first_call:
            self.n_features_ = X.shape[1]
            self.n_classes_ = np.array(n_classes, dtype=np.intp)
            self.n_outputs_ = 1
            self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)
            # Fit the categorical encoder once on the first batch; later
            # batches map through the frozen statistics.
            encoder = self._fit_categorical_encoder(X, y[:, 0])
            self._cat_encoder_ = (
                encoder if encoder is not None and encoder.active_ else None)

        if getattr(self, "_cat_encoder_", None) is not None:
            X = self._cat_encoder_.transform(X)

        builder = PartialFitTreeBuilder(
            self.min_samples_split, max_depth, random_state)
        builder.build(self.tree_, X, y)
        return self

    def weighted_decision_path(self, X, check_input=True):
        """
        Returns the weighted decision path in the tree.

        Each non-zero value in the decision path determines the weight
        of that particular node in making predictions.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicate the weight of that particular node in making predictions.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.weighted_decision_path(X)


class MondrianTreeRegressor(BaseMondrianTree, RegressorMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 random_state=None,
                 path_smoothing=False,
                 categorical_features="auto",
                 max_categories=32):
        super(MondrianTreeRegressor, self).__init__(
            criterion="mse",
            splitter="mondrian",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            path_smoothing=path_smoothing,
            categorical_features=categorical_features,
            max_categories=max_categories)

    @classmethod
    def from_model(cls, model, X, y):
        """Create a MondrianTreeRegressor from a fitted sklearn or ONNX model.

        This classmethod converts a fitted gradient boosting or random forest
        model (provided as a sklearn estimator or ONNX model) into a
        MondrianTreeRegressor. The conversion preserves the original model's
        predictions while rebuilding Mondrian-specific statistics (bounds,
        tau values, node samples) needed for incremental training via
        partial_fit.

        Parameters
        ----------
        model : sklearn estimator or ONNX ModelProto
            A fitted sklearn tree/forest estimator (with ``tree_`` or
            ``estimators_`` attribute) or an ONNX model proto.
        X : array-like of shape (n_samples, n_features)
            Training data used to compute Mondrian statistics. Should have at
            least 300 samples for meaningful statistics.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        MondrianTreeRegressor
            A fitted Mondrian tree that produces the same predictions as the
            original model and supports partial_fit.

        Raises
        ------
        ValueError
            If the model is not a valid tree model.
        ImportError
            If onnx is required but not installed.

        Warnings
        --------
        UserWarning
            Issued if the training dataset has fewer than 300 samples.

        Examples
        --------
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from shinrin import MondrianTreeRegressor
        >>> from shinrin.onnx import to_onnx
        >>>
        >>> sklearn_model = GradientBoostingRegressor(n_estimators=1, max_depth=3)
        >>> sklearn_model.fit(X_train, y_train)
        >>> onnx_model = to_onnx(sklearn_model, X_train)
        >>>
        >>> # Convert to Mondrian tree (one-step)
        >>> tree = MondrianTreeRegressor.from_model(onnx_model, X_train, y_train)
        >>> tree.predict(X_test)  # same predictions as sklearn_model
        >>> tree.partial_fit(X_new, y_new)  # continue training
        """
        from shinrin.onnx_import import from_model

        return from_model(model, X, y, cls)

    def partial_fit(self, X, y):
        """
        Incremental building of Mondrian Tree Regressors.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        Returns
        -------
        self: instance of MondrianTree
        """
        return super(MondrianTreeRegressor, self).partial_fit(X, y)

class MondrianTreeClassifier(BaseMondrianTree, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 random_state=None,
                 path_smoothing=False,
                 categorical_features="auto",
                 max_categories=32):
        super(MondrianTreeClassifier, self).__init__(
            criterion="classification",
            splitter="mondrian",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            path_smoothing=path_smoothing,
            categorical_features=categorical_features,
            max_categories=max_categories)

    @classmethod
    def from_model(cls, model, X, y):
        """Create a MondrianTreeClassifier from a fitted sklearn or ONNX model.

        This classmethod converts a fitted gradient boosting or random forest
        classifier (provided as a sklearn estimator or ONNX model) into a
        MondrianTreeClassifier. The conversion preserves the original model's
        predictions while rebuilding Mondrian-specific statistics (bounds,
        tau values, node samples) needed for incremental training via
        partial_fit.

        Parameters
        ----------
        model : sklearn estimator or ONNX ModelProto
            A fitted sklearn tree/forest estimator (with ``tree_`` or
            ``estimators_`` attribute) or an ONNX model proto.
        X : array-like of shape (n_samples, n_features)
            Training data used to compute Mondrian statistics. Should have at
            least 300 samples for meaningful statistics.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        MondrianTreeClassifier
            A fitted Mondrian tree that produces the same predictions as the
            original model and supports partial_fit.

        Raises
        ------
        ValueError
            If the model is not a valid tree model.
        ImportError
            If onnx is required but not installed.

        Warnings
        --------
        UserWarning
            Issued if the training dataset has fewer than 300 samples.

        Examples
        --------
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from shinrin import MondrianTreeClassifier
        >>> from shinrin.onnx import to_onnx
        >>>
        >>> sklearn_model = GradientBoostingClassifier(n_estimators=1, max_depth=3)
        >>> sklearn_model.fit(X_train, y_train)
        >>> onnx_model = to_onnx(sklearn_model, X_train)
        >>>
        >>> # Convert to Mondrian tree (one-step)
        >>> tree = MondrianTreeClassifier.from_model(onnx_model, X_train, y_train)
        >>> tree.predict_proba(X_test)  # same predictions as sklearn_model
        >>> tree.partial_fit(X_new, y_new)  # continue training
        """
        from shinrin.onnx_import import from_model

        return from_model(model, X, y, cls)

    def predict_proba(self, X, check_input=True, path_smoothing=None):
        """
        Predicts the probability of each class label given X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        path_smoothing : boolean, optional
            Override the estimator's ``path_smoothing`` setting for this
            call. ``None`` (default) uses the value chosen at construction
            time. See the class docstring for what the mode means.

        Returns
        -------
        y_prob : array of shape = [n_samples, n_classes]
            Prediceted probabilities for each class.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)

        return self.tree_.predict(
            X, return_std=False, is_regression=False,
            path_smoothing=self._resolve_path_smoothing(path_smoothing))[0]

    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Tree Classifiers.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianTree
        """
        return super(MondrianTreeClassifier, self).partial_fit(
            X, y, classes=classes)
