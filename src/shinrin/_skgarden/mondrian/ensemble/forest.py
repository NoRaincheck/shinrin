import numpy as np
import threading
from scipy import sparse
from shinrin._compat.sklearn_base import ClassifierMixin
from shinrin._compat.sklearn_exceptions import DataConversionWarning
from shinrin._compat.sklearn_exceptions import NotFittedError
from shinrin._compat.sklearn_preprocessing import LabelEncoder
from shinrin._compat.sklearn_utils import check_random_state
from shinrin._compat.sklearn_utils_validation import check_array
from shinrin._compat.sklearn_utils_validation import check_is_fitted
from shinrin._compat.sklearn_utils_validation import check_X_y
from joblib import delayed, Parallel

from warnings import warn


def _c_n(n):
    """Average path length of unsuccessful search in a BST with n nodes.

    Used to normalize isolation forest anomaly scores.
    c(n) = 2*H(n-1) - 2*(n-1)/n where H(i) is the harmonic number.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # H(n-1) ≈ ln(n-1) + 0.5772156649 (Euler-Mascheroni constant)
    from math import log, e
    h_n = log(n - 1) + 0.5772156649
    return 2.0 * h_n - 2.0 * (n - 1) / n


from ..tree import MondrianTreeClassifier
from ..tree import MondrianTreeRegressor

from ...forest import ForestClassifier
from ...forest import ForestRegressor
from ...forest import _accumulate_prediction
from ...forest import _joblib_parallel_args
from ...forest import _partition_estimators

def _single_tree_pfit(tree, X, y, classes=None):
    if classes is not None:
        tree.partial_fit(X, y, classes)
    else:
        tree.partial_fit(X, y)
    return tree

class BaseMondrian(object):
    def pred_contribs(self, X, path_smoothing=None):
        """Return TreeSHAP values including the base value.

        This method returns SHAP values averaged across all trees such that
        the sum of SHAP values plus the base value equals the model
        prediction under the effective ``path_smoothing`` mode.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input samples.

        path_smoothing : boolean, optional
            Override the forest's ``path_smoothing`` setting for this call.
            ``None`` (default) uses the value chosen at construction time.

        Returns
        -------
        shap_values : array
            The last column/axis contains the base value (root prediction).
            For regression, shape is (n_samples, n_features + 1).
            For classification with K classes, shape is
            (n_samples, n_features + 1, K).
        """
        smoothing = self._resolve_path_smoothing(path_smoothing)
        X = self._validate_X_predict(X)
        n_samples = X.shape[0]
        n_features = self.n_features_

        ensemble_shap = None
        base_values = None
        is_classification = None

        for est in self.estimators_:
            tree_contribs = est.pred_contribs(X, path_smoothing=smoothing)
            # tree_contribs shape: (n_samples, n_features + 1) for reg
            # or (n_samples, n_features + 1, K) for clf
            if isinstance(est, ClassifierMixin):
                if is_classification is None:
                    is_classification = True
                n_classes = tree_contribs.shape[-1]
                if ensemble_shap is None:
                    ensemble_shap = np.zeros((n_samples, n_features, n_classes))
                    base_values = np.zeros((n_samples, n_classes))
                ensemble_shap += tree_contribs[:, :-1, :]
                base_values += tree_contribs[:, -1, :]
            else:
                if is_classification is None:
                    is_classification = False
                if ensemble_shap is None:
                    ensemble_shap = np.zeros((n_samples, n_features + 1))
                    base_values = np.zeros((n_samples,))
                ensemble_shap += tree_contribs
                base_values += tree_contribs[:, -1]

        ensemble_shap /= len(self.estimators_)
        base_values /= len(self.estimators_)

        if is_classification:
            # ensemble_shap: (n_samples, n_features, n_classes)
            # base_values: (n_samples, n_classes)
            # Result: (n_samples, n_features + 1, n_classes)
            return np.concatenate(
                [ensemble_shap, base_values[:, np.newaxis, :]], axis=1
            )
        else:
            return np.column_stack([ensemble_shap[:, :-1], base_values])

    def pred_anomaly(self, X, n_train=None):
        """Compute Isolation Forest anomaly scores.

        Each sample receives an anomaly score based on its average path length
        through the forest. Scores range from 0 to 1, where:
            - ~0.5: normal sample (path length ~ average)
            - ~1.0: highly anomalous (short path length)
            - ~0.0: very deep path (unlikely under isolation model)

        The score is computed as:
            s(x, n) = 2^(-E[h(x)] / c(n))

        where h(x) is the path length from root to leaf and c(n) is the
        average path length of an unsuccessful search in a binary search tree
        with n nodes.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input samples.

        n_train : int, optional
            Number of training samples used to build the forest. If not
            provided, inferred from the first tree's n_node_samples.
            Using the correct n_train is important for proper normalization.

        Returns
        -------
        anomaly_scores : array-like, shape = (n_samples,)
            Anomaly scores for each sample. Higher values indicate more
            anomalous samples.
        """
        X = self._validate_X_predict(X)
        total_anomaly = np.zeros(X.shape[0])
        for est in self.estimators_:
            total_anomaly += est._compute_anomaly(X)

        avg_anomaly = total_anomaly / len(self.estimators_)

        # Normalize by c(n) - use training sample count for proper normalization
        if n_train is None:
            n_train = len(self.estimators_[0].tree_.n_node_samples)
        c_n = _c_n(n_train)
        if c_n == 0:
            return np.zeros(X.shape[0])
        return np.power(2, -avg_anomaly / c_n)
        """Return TreeSHAP values including the base value.

        This method returns SHAP values averaged across all trees such that
        the sum of SHAP values plus the base value equals the model prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input samples.

        Returns
        -------
        shap_values : array of shape = (n_samples, n_features + 1)
            The last column contains the base value (root prediction).
            For regression, shape is (n_samples, n_features + 1).
            For classification with K classes, shape is
            (n_samples, n_features + 1, K).
        """
        X = self._validate_X_predict(X)
        n_samples = X.shape[0]
        n_features = self.n_features_

        ensemble_shap = np.zeros((n_samples, n_features))
        base_values = np.zeros((n_samples,))

        for est in self.estimators_:
            tree_contribs = est.pred_contribs(X)
            # tree_contribs shape: (n_samples, n_features + 1) for reg
            # or (n_samples, n_features + 1, K) for clf
            if isinstance(est, ClassifierMixin):
                ensemble_shap += tree_contribs[:, :-1, :].sum(axis=-1)
                base_values += tree_contribs[:, -1, :].mean(axis=-1)
            else:
                ensemble_shap += tree_contribs[:, :-1]
                base_values += tree_contribs[:, -1]

        ensemble_shap /= len(self.estimators_)
        base_values /= len(self.estimators_)

        return np.column_stack([ensemble_shap, base_values])

    def weighted_decision_path(self, X):
        """
        Returns the weighted decision path in the forest.

        Each non-zero value in the decision path determines the
        weight of that particular node while making predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input.

        Returns
        -------
        decision_path : sparse csr matrix, shape = (n_samples, n_total_nodes)
            Return a node indicator matrix where non zero elements
            indicate the weight of that particular node in making predictions.

        est_inds : array-like, shape = (n_estimators + 1,)
            weighted_decision_path[:, est_inds[i]: est_inds[i + 1]]
            provides the weighted_decision_path of estimator i
        """
        X = self._validate_X_predict(X)
        est_inds = np.cumsum(
            [0] + [est.tree_.node_count for est in self.estimators_])
        paths = sparse.hstack(
            [est.weighted_decision_path(X) for est in self.estimators_]).tocsr()
        return paths, est_inds

    # XXX: This is mainly a stripped version of BaseForest.fit
    # from sklearn.forest
    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Forests.

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
        self: instance of MondrianForest
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        random_state = check_random_state(self.random_state)

        # Wipe out estimators if partial_fit is called after fit.
        first_call = not hasattr(self, "first_")
        if first_call:
            self.first_ = True

        if isinstance(self, ClassifierMixin):
            if first_call:
                if classes is None:
                    classes = LabelEncoder().fit(y).classes_

                self.classes_ = classes
                self.n_classes_ = len(self.classes_)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        self.n_outputs_ = 1

        # Initialize estimators at first call to partial_fit.
        if first_call:
            # Check estimators
            self._validate_estimator()
            self.estimators_ = []

            for _ in range(self.n_estimators):
                tree = self._make_estimator(append=False, random_state=random_state)
                self.estimators_.append(tree)

        # XXX: Switch to threading backend when GIL is released.
        # Disable parallelism for ONNX-converted forests (PyTree is unsendable)
        effective_n_jobs = 1 if hasattr(self, '_onnx_converted') else self.n_jobs
        if isinstance(self, ClassifierMixin):
            self.estimators_ = Parallel(n_jobs=effective_n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y, classes) for t in self.estimators_)
        else:
            self.estimators_ = Parallel(n_jobs=effective_n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y) for t in self.estimators_)

        return self


class MondrianForestRegressor(ForestRegressor, BaseMondrian):
    """
    A MondrianForestRegressor is an ensemble of MondrianTreeRegressors.

    The variance in predictions is reduced by averaging the predictions
    from all trees.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_depth : integer, optional (default=None)
        The depth to which each tree is grown. If None, the tree is either
        grown to full depth or is constrained by `min_samples_split`.

    min_samples_split : integer, optional (default=2)
        Stop growing the tree if all the nodes have lesser than
        `min_samples_split` number of samples.

    bootstrap : boolean, optional (default=False)
        If bootstrap is set to False, then all trees are trained on the
        entire training dataset. Else, each tree is fit on n_samples
        drawn with replacement from the training dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    path_smoothing : bool, optional (default=False)
        Prediction mode for every tree in the forest. With the default
        ``False``, predictions are piecewise-constant leaf values averaged
        across trees (scikit-learn-consistent, and exactly what the plain
        ONNX ``ai.onnx.ml`` tree-ensemble export computes). This is an
        *opinionated default* that deviates from the "pure" Mondrian-process
        prediction; with ``True``, each tree weights every node on its
        decision path (see ``MondrianTreeRegressor``). SHAP contributions
        and anomaly scores always use the Mondrian node weights regardless
        of this setting.
    """
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 path_smoothing=False):
        super(MondrianForestRegressor, self).__init__(
            base_estimator=MondrianTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("max_depth", "min_samples_split",
                              "random_state", "path_smoothing"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.path_smoothing = path_smoothing

    def _resolve_path_smoothing(self, path_smoothing):
        """Resolve the effective prediction mode for a predict-time call."""
        if path_smoothing is None:
            return bool(getattr(self, "path_smoothing", False))
        return bool(path_smoothing)

    @classmethod
    def from_model(cls, model, X, y):
        """Create a MondrianForestRegressor from a fitted sklearn or ONNX model.

        This classmethod converts a fitted gradient boosting or random forest
        model (provided as a sklearn estimator or ONNX model) into a
        MondrianForestRegressor. The conversion preserves the original model's
        predictions while rebuilding Mondrian-specific statistics (bounds,
        tau values, node samples) needed for incremental training via
        partial_fit.

        Parameters
        ----------
        model : sklearn estimator or ONNX ModelProto
            A fitted sklearn forest/boosting estimator (with ``estimators_``
            attribute) or an ONNX model proto.
        X : array-like of shape (n_samples, n_features)
            Training data used to compute Mondrian statistics. Should have at
            least 300 samples for meaningful statistics.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        MondrianForestRegressor
            A fitted Mondrian forest that produces the same predictions as
            the original model and supports partial_fit.

        Raises
        ------
        ValueError
            If the model is not a valid forest model.
        ImportError
            If onnx is required but not installed.

        Warnings
        --------
        UserWarning
            Issued if the training dataset has fewer than 300 samples.

        Examples
        --------
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from shinrin import MondrianForestRegressor
        >>> from shinrin.onnx import to_onnx
        >>>
        >>> sklearn_model = GradientBoostingRegressor(n_estimators=10, max_depth=3)
        >>> sklearn_model.fit(X_train, y_train)
        >>> onnx_model = to_onnx(sklearn_model, X_train)
        >>>
        >>> # Convert to Mondrian forest (one-step)
        >>> forest = MondrianForestRegressor.from_model(
        ...     onnx_model, X_train, y_train
        ... )
        >>> forest.predict(X_test)  # same predictions as sklearn_model
        >>> forest.partial_fit(X_new, y_new)  # continue training
        """
        from shinrin.onnx_import import from_model

        return from_model(model, X, y, cls)

    def fit(self, X, y):
        """Builds a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        return super(MondrianForestRegressor, self).fit(X, y)

    def predict(self, X, return_std=False, return_anomaly=False,
                return_shap=False, path_smoothing=None):
        """
        Returns the predicted mean and std.

        The prediction is a GMM drawn from
        ``\\sum_{i=1}^T w_i N(m_i, \\sigma_i)`` where ``w_i = {1 \\over T}``.

        The mean ``E[Y | X]`` reduces to ``{\\sum_{i=1}^T m_i \\over T}``

        The variance ``Var[Y | X]`` is given by
        ``Var[Y | X] = E[Y^2 | X] - E[Y | X]^2``
        ``= \\frac{\\sum_{i=1}^T E[Y^2_i| X]}{T} - E[Y | X]^2``
        $$= \\frac{\\sum_{i=1}^T (Var[Y_i | X] + E[Y_i | X]^2)}{T} - E[Y| X]^2$$

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input samples.

        return_std : boolean, default (False)
            Whether or not to return the standard deviation.

        return_anomaly : boolean, default (False)
            If True, return the Isolation Forest anomaly score for each sample.
            The anomaly score is computed as:
                s(x, n) = 2^(-E[h(x)] / c(n))
            where h(x) is the path length and c(n) is the average path length
            of unsuccessful search in a BST. Scores near 1 indicate anomalies.

        return_shap : boolean, default (False)
            If True, return TreeSHAP values averaged across all trees.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            Predictions at X.

        std : array-like, shape = (n_samples,), optional
            Standard deviation at X. Returned if ``return_std=True``.

        anomaly_scores : array-like, shape = (n_samples,), optional
            Isolation Forest anomaly scores. Returned if ``return_anomaly=True``.

        shap_values : array-like, shape = (n_samples, n_features), optional
            TreeSHAP values averaged across all trees. Returned if
            ``return_shap=True``.

        path_smoothing : boolean, optional
            Override the forest's ``path_smoothing`` setting for this call.
            ``None`` (default) uses the value chosen at construction time.
        """
        X = check_array(X)
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model has to be fit before prediction.")
        smoothing = self._resolve_path_smoothing(path_smoothing)
        ensemble_mean = np.zeros(X.shape[0])
        exp_y_sq = np.zeros_like(ensemble_mean)
        ensemble_anomaly = np.zeros(X.shape[0])
        ensemble_shap = np.zeros((X.shape[0], self.n_features_))

        for est in self.estimators_:
            # Compute anomaly and shap from tree directly
            tree_anomaly = est._compute_anomaly(X) if return_anomaly else None
            tree_shap = (est._compute_shap(X, path_smoothing=smoothing)
                         if return_shap else None)

            if return_std:
                mean, std = est.predict(X, return_std=True,
                                        path_smoothing=smoothing)
                exp_y_sq += (std**2 + mean**2)
            else:
                mean = est.predict(X, return_std=False,
                                   path_smoothing=smoothing)
            ensemble_mean += mean

            if return_anomaly:
                ensemble_anomaly += tree_anomaly

            if return_shap:
                ensemble_shap += tree_shap

        ensemble_mean /= len(self.estimators_)
        exp_y_sq /= len(self.estimators_)

        if return_anomaly:
            ensemble_anomaly /= len(self.estimators_)
            # Normalize by c(n) = average path length of unsuccessful BST search
            # Use training sample count for proper normalization
            n_train = len(self.estimators_[0].tree_.n_node_samples)
            c_n = _c_n(n_train)
            anomaly_scores = np.power(2, -ensemble_anomaly / c_n) if c_n > 0 else np.zeros(X.shape[0])

        if return_shap:
            ensemble_shap /= len(self.estimators_)

        # For ONNX-converted GradientBoosting, sum predictions instead of averaging
        # (base_values is already included in the first tree's predictions)
        if hasattr(self, "_onnx_sum_predictions"):
            ensemble_mean = ensemble_mean * len(self.estimators_)
            if return_shap:
                ensemble_shap = ensemble_shap * len(self.estimators_)

        if not return_std and not return_anomaly and not return_shap:
            return ensemble_mean

        results = [ensemble_mean]
        if return_std:
            std = exp_y_sq - ensemble_mean**2
            std[std <= 0.0] = 0.0
            std **= 0.5
            results.append(std)
        if return_anomaly:
            results.append(anomaly_scores)
        if return_shap:
            results.append(ensemble_shap)

        return tuple(results)

    def partial_fit(self, X, y):
        """
        Incremental building of Mondrian Forest Regressors.

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
        self: instance of MondrianForestClassifier
        """
        return super(MondrianForestRegressor, self).partial_fit(X, y)


class MondrianForestClassifier(ForestClassifier, BaseMondrian):
    """
    A MondrianForestClassifier is an ensemble of MondrianTreeClassifiers.

    The probability ``p_j`` of class ``j`` is given by the average
    across all trees: ``\\sum_{i}^{N_{est}} \\frac{p_{j}^i}{N_{est}}``

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_depth : integer, optional (default=None)
        The depth to which each tree is grown. If None, the tree is either
        grown to full depth or is constrained by `min_samples_split`.

    min_samples_split : integer, optional (default=2)
        Stop growing the tree if all the nodes have lesser than
        `min_samples_split` number of samples.

    bootstrap : boolean, optional (default=False)
        If bootstrap is set to False, then all trees are trained on the
        entire training dataset. Else, each tree is fit on n_samples
        drawn with replacement from the training dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    path_smoothing : bool, optional (default=False)
        Prediction mode for every tree in the forest. With the default
        ``False``, class probabilities are the per-leaf class distributions
        averaged across trees (scikit-learn-consistent). This is an
        *opinionated default* that deviates from the "pure" Mondrian-process
        prediction; with ``True``, each tree weights every node on its
        decision path (see ``MondrianTreeClassifier``).
    """
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 path_smoothing=False):
        super(MondrianForestClassifier, self).__init__(
            base_estimator=MondrianTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("max_depth", "min_samples_split",
                              "random_state", "path_smoothing"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.path_smoothing = path_smoothing

    def _resolve_path_smoothing(self, path_smoothing):
        """Resolve the effective prediction mode for a predict-time call."""
        if path_smoothing is None:
            return bool(getattr(self, "path_smoothing", False))
        return bool(path_smoothing)

    @classmethod
    def from_model(cls, model, X, y):
        """Create a MondrianForestClassifier from a fitted sklearn or ONNX model.

        This classmethod converts a fitted gradient boosting or random forest
        classifier (provided as a sklearn estimator or ONNX model) into a
        MondrianForestClassifier. The conversion preserves the original model's
        predictions while rebuilding Mondrian-specific statistics (bounds,
        tau values, node samples) needed for incremental training via
        partial_fit.

        Parameters
        ----------
        model : sklearn estimator or ONNX ModelProto
            A fitted sklearn forest/boosting estimator (with ``estimators_``
            attribute) or an ONNX model proto.
        X : array-like of shape (n_samples, n_features)
            Training data used to compute Mondrian statistics. Should have at
            least 300 samples for meaningful statistics.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        MondrianForestClassifier
            A fitted Mondrian forest that produces the same predictions as
            the original model and supports partial_fit.

        Raises
        ------
        ValueError
            If the model is not a valid forest model.
        ImportError
            If onnx is required but not installed.

        Warnings
        --------
        UserWarning
            Issued if the training dataset has fewer than 300 samples.

        Examples
        --------
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from shinrin import MondrianForestClassifier
        >>> from shinrin.onnx import to_onnx
        >>>
        >>> sklearn_model = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        >>> sklearn_model.fit(X_train, y_train)
        >>> onnx_model = to_onnx(sklearn_model, X_train)
        >>>
        >>> # Convert to Mondrian forest (one-step)
        >>> forest = MondrianForestClassifier.from_model(
        ...     onnx_model, X_train, y_train
        ... )
        >>> forest.predict(X_test)  # same predictions as sklearn_model
        >>> forest.partial_fit(X_new, y_new)  # continue training
        """
        from shinrin.onnx_import import from_model

        return from_model(model, X, y, cls)

    def fit(self, X, y):
        """Builds a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        return super(MondrianForestClassifier, self).fit(X, y)

    def predict(self, X, return_anomaly=False, return_shap=False,
                check_input=True, path_smoothing=None):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = (n_samples, n_features)
            The input samples.

        return_anomaly : boolean, default (False)
            If True, return the Isolation Forest anomaly score alongside
            the predicted class.

        return_shap : boolean, default (False)
            If True, return TreeSHAP values averaged across all trees.

        check_input : boolean, default (True)
            Allow to bypass several input checking.

        path_smoothing : boolean, optional
            Override the forest's ``path_smoothing`` setting for this call.
            ``None`` (default) uses the value chosen at construction time.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            The predicted classes.

        anomaly_scores : array-like, shape = (n_samples,), optional
            Isolation Forest anomaly scores. Returned if
            ``return_anomaly=True``.

        shap_values : array-like, shape = (n_samples, n_features), optional
            TreeSHAP values averaged across all trees. Returned if
            ``return_shap=True``.
        """
        proba = self.predict_proba(X, path_smoothing=path_smoothing)
        predictions = self.classes_.take(np.argmax(proba, axis=1), axis=0)

        results = [predictions]

        if return_anomaly:
            results.append(self.pred_anomaly(X))

        if return_shap:
            n_classes = self.n_classes_
            ensemble_shap = np.zeros((X.shape[0], self.n_features_, n_classes))
            for est in self.estimators_:
                ensemble_shap += est._compute_shap(
                    X, path_smoothing=self._resolve_path_smoothing(
                        path_smoothing))
            ensemble_shap /= len(self.estimators_)
            results.append(ensemble_shap)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Forest Classifiers.

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
        self: instance of MondrianForestClassifier
        """
        return super(MondrianForestClassifier, self).partial_fit(
            X, y, classes=classes)

    def predict_proba(self, X, path_smoothing=None):
        """Predict class probabilities for X.

        Probabilities are the mean across all trees.

        Parameters
        ----------
        X : array-like of shape = (n_samples, n_features)
            The input samples.

        path_smoothing : boolean, optional
            Override the forest's ``path_smoothing`` setting for this call.
            ``None`` (default) uses the value chosen at construction time.

        Returns
        -------
        p : array of shape = (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        smoothing = self._resolve_path_smoothing(path_smoothing)

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # If first_ is True, the forest was imported from sklearn/ONNX
        # and trees are unsendable, so disable parallelism
        if getattr(self, 'first_', False):
            n_jobs = 1

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(
                lambda x, check_input=False, est=est: est.predict_proba(
                    x, check_input=check_input, path_smoothing=smoothing),
                X, all_proba, lock)
            for est in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        return all_proba
