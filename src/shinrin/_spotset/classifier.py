"""SPOTSET classifier — Rashomon sets of sparse decision trees.

SPOTSET (**S**parse **O**ptimal **R**ashomon **SET**) enumerates the set of
near-optimal sparse decision trees whose regularized loss is within a
configurable bound of the optimum, instead of returning a single optimal
tree. Renamed from treeFARMS ("Trees FAst RashoMon Sets",
https://github.com/ubc-systopia/treeFARMS, BSD-3-Clause); see README and
NOTICE for provenance.

Features should already be binary/binarized (e.g. via
:class:`shinrin.SpotThresholdGuessBinarizer` or :class:`shinrin.NumericBinarizer`);
the engine's own encoder additionally handles integral/rational/categorical
columns, but prediction operates in the encoded feature space, so binarized
inputs keep fit/predict consistent.
"""

import io
import json
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from shinrin._native import spotset_configure, spotset_fit

from .model_set import ModelSetContainer


def _format_value(v) -> str:
    """Format a feature value for the engine's CSV parser (no scientific
    notation, integers kept integral)."""
    fv = float(v)
    if fv.is_integer():
        return str(int(fv))
    return f"{fv:.12f}"


class SPOTSETClassifier(ClassifierMixin, BaseEstimator):
    """Classifier extracting the Rashomon set of sparse decision trees.

    Renamed from treeFARMS' ``TREEFARMS``; SPOTSET stands for Sparse Optimal
    Rashomon Trees. After ``fit``, the extracted set is available as
    ``self.model_set_`` (a :class:`~shinrin._spotset.model_set.ModelSetContainer`);
    ``self[i]`` decodes the i-th tree of the set.

    Parameters
    ----------
    regularization : float, default=0.05
        Complexity penalty per leaf added to the misclassification loss.
        Recommended: larger than ``1 / n_samples``.
    rashomon_bound_multiplier : float, default=0.05
        Size of the explored set: the Rashomon bound is
        ``(1 + multiplier) * optimal objective``. Larger values explore
        exponentially more trees.
    rashomon : bool, default=True
        Extract the full Rashomon set. If False, only near-optimal search
        bounds are used and a single-model set is returned.
    depth_budget : int | None, default=None
        Maximum tree depth (root-only tree = depth 1); ``None`` means unlimited.
    time_limit : int | None, default=None
        Seconds; on timeout training stops with status 2 and the partial set.
    worker_limit : int, default=1
        Number of parallel search workers; ``0`` uses one worker per core.
    verbose : bool, default=False
        Print engine progress information.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique classes seen during fit.
    model_set_ : ModelSetContainer
        The extracted Rashomon set.
    n_trees_ : int
        Number of trees in the extracted set.
    train_time_ : float
        Seconds spent in the native search.

    Examples
    --------
    >>> from shinrin import ThresholdGuessBinarizer  # doctest: +SKIP
    >>> from shinrin._spotset import SPOTSETClassifier
    >>> X_bin = ThresholdGuessBinarizer().fit_transform(X, y)
    >>> clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.05)
    >>> clf.fit(X_bin, y)
    >>> clf.n_trees_
    """

    def __init__(
        self,
        regularization: float = 0.05,
        rashomon_bound_multiplier: float = 0.05,
        rashomon: bool = True,
        depth_budget=None,
        time_limit=None,
        worker_limit: int = 1,
        verbose: bool = False,
    ):
        self.regularization = regularization
        self.rashomon_bound_multiplier = rashomon_bound_multiplier
        self.rashomon = rashomon
        self.depth_budget = depth_budget
        self.time_limit = time_limit
        self.worker_limit = worker_limit
        self.verbose = verbose

    def _configuration(self) -> dict:
        worker_limit = self.worker_limit
        if worker_limit == 0:
            worker_limit = os.cpu_count() or 1
        configuration = {
            "regularization": float(self.regularization),
            "rashomon": bool(self.rashomon),
            "rashomon_bound_multiplier": float(self.rashomon_bound_multiplier),
            "worker_limit": int(worker_limit),
            "verbose": bool(self.verbose),
            "cancelation": True,  # sic: upstream key spelling
            "look_ahead": True,
            "similar_support": True,
        }
        if self.depth_budget is not None:
            configuration["depth_budget"] = int(self.depth_budget)
        if self.time_limit is not None:
            configuration["time_limit"] = int(self.time_limit)
        return configuration

    def _dataset_csv(self, X: np.ndarray, y_indices: np.ndarray) -> str:
        buffer = io.StringIO()
        n, m = X.shape
        header = ",".join([f"f{j}" for j in range(m)] + ["class"])
        buffer.write(header + "\n")
        for i in range(n):
            row = ",".join([_format_value(v) for v in X[i]] + [str(int(y_indices[i]))])
            buffer.write(row + "\n")
        return buffer.getvalue()

    def fit(self, X, y):
        """Extract the Rashomon set of sparse decision trees for (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary features; binarize continuous features first (e.g. with
            ``ThresholdGuessBinarizer``).
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        label_encoder = LabelBinarizer()
        y_indices = label_encoder.fit_transform(y)
        if y_indices.shape[1] == 1:
            # binary case: LabelBinarizer emits a single 0/1 column
            y_column = y_indices[:, 0]
        else:
            y_column = np.argmax(y_indices, axis=1)
        self.classes_ = label_encoder.classes_

        spotset_configure(json.dumps(self._configuration(), separators=(",", ":")))
        result = spotset_fit(self._dataset_csv(X, y_column))

        status = result["status"]
        if status == 2:
            import warnings

            warnings.warn(
                "SPOTSET reported a possible timeout; returning the partial "
                "Rashomon set extracted so far.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif status != 0:
            raise RuntimeError(
                f"SPOTSET encountered an error while training (status {status})"
            )

        self.model_set_ = ModelSetContainer(json.loads(result["model"]))
        self.n_trees_ = self.model_set_.get_tree_count()
        self.train_time_ = float(result["time"])
        self.n_features_in_ = X.shape[1]
        self._tree_cache = None
        return self

    def _tree(self):
        if getattr(self, "_tree_cache", None) is None:
            self._tree_cache = self.model_set_.get_tree_at_idx(0)
        return self._tree_cache

    def predict(self, X):
        """Predict class labels using the first (lowest-objective) tree of the set."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        y_index = self._tree().predict(X).astype(int)
        return self.classes_[y_index]

    def get_tree_count(self):
        """Number of trees in the extracted Rashomon set."""
        check_is_fitted(self)
        return self.n_trees_

    def __getitem__(self, idx):
        """Obtain the ``idx``-th tree of the set as a ``TreeClassifier``.

        Note: trees predict integer-encoded labels; map back through
        ``clf.classes_``.
        """
        check_is_fitted(self)
        return self.model_set_.get_tree_at_idx(idx)

    def get_decision_paths(self):
        """Hierarchical trie describing all decision paths in the Rashomon set."""
        check_is_fitted(self)
        return self.model_set_.to_trie()

    def __repr__(self, n_char_max=400):
        return super().__repr__(n_char_max=n_char_max)
