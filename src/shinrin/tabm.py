"""Scikit-learn compatible interface for the vendored TabM model.

Provides :class:`TabMClassifier` and :class:`TabMRegressor`, opinionated
drop-in replacements for ``sklearn.neural_network.MLPClassifier`` /
``MLPRegressor`` powered by TabM (parameter-efficient ensembling of MLPs,
ICLR 2025). Training runs entirely on NumPy, optionally accelerated by
the bundled Mojo kernels — PyTorch is never required.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data

from shinrin._tabm import _backend as tabm_backend
from shinrin._quant import validate_quantization
from shinrin._tabm._layers import TabMConfig, TabMParams
from shinrin._tabm._model import Batch, TabMCore
from shinrin._tabm._optim import (
    AdamState,
    FlatSpace,
    adam_step,
    lbfgs_minimize,
    sgd_step,
)
from shinrin._tabm._transforms import (
    AsinhTransform,
    PiecewiseLinearEncoder,
    QuantileTransform,
    StandardScalerTransform,
    build_num_bins,
    detect_categorical_features,
)

__all__ = ["TabMClassifier", "TabMRegressor"]


class _Preprocessor:
    """Fitted preprocessing pipeline (transforms + encodings)."""

    def __init__(self, base: _BaseTabM) -> None:
        self.base = base

    def fit(self, X: np.ndarray) -> _Preprocessor:
        p = self.base
        self.categorical_indices_, self.numerical_indices_, self.cardinalities_ = (
            detect_categorical_features(
                X,
                cardinality_threshold=p.categorical_cardinality_threshold,
                categorical_indices=p.categorical_indices,
            )
        )
        num_idx = self.numerical_indices_
        self.transforms_ = []
        if num_idx:
            X_num = X[:, num_idx]
            if p.use_quantile:
                t = QuantileTransform(num_quantiles=100).fit(X_num)
                X_num = t.transform(X_num)
                self.transforms_.append(t)
            if p.use_asinh:
                self.transforms_.append(AsinhTransform().fit(X_num))
                X_num = self.transforms_[-1].transform(X_num)
            if p.use_scaler:
                t = StandardScalerTransform().fit(X_num)
                X_num = t.transform(X_num)
                self.transforms_.append(t)
            self.bins_ = build_num_bins(X_num, p.n_bins)
            self.encoder_ = PiecewiseLinearEncoder(self.bins_)
        else:
            self.bins_ = []
            self.encoder_ = None

        self.value_maps_: list[dict[float, int]] = []
        for col in self.categorical_indices_:
            values = X[:, col]
            values = values[~np.isnan(values)]
            unique = np.unique(values)
            self.value_maps_.append({float(v): i for i, v in enumerate(unique)})
        return self

    @property
    def d_enc(self) -> int:
        return self.encoder_.width if self.encoder_ is not None else 0

    @property
    def d_cat(self) -> int:
        return sum(self.cardinalities_)

    def transform(
        self, X: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return ``(x_num, x_enc, x_cat)`` float32 arrays."""
        p = self.base
        num_idx = self.numerical_indices_
        x_num: np.ndarray | None = None
        x_enc: np.ndarray | None = None
        if num_idx:
            x_num = np.ascontiguousarray(X[:, num_idx], dtype=np.float32)
            current = x_num
            for t in self.transforms_:
                current = t.transform(current)
            if self.encoder_ is not None and p.use_embeddings:
                x_enc = np.ascontiguousarray(
                    self.encoder_.transform(current), dtype=np.float32
                )
        cat_parts = []
        for block, mapping in zip(self.categorical_indices_, self.value_maps_):
            col = X[:, block]
            idx = np.array([mapping.get(float(v), 0) for v in col], dtype=np.int64)
            onehot = np.zeros((len(col), len(mapping)), dtype=np.float32)
            onehot[np.arange(len(col)), idx] = 1.0
            cat_parts.append(onehot)
        x_cat = (
            np.ascontiguousarray(np.concatenate(cat_parts, axis=1))
            if cat_parts
            else None
        )
        return x_num, x_enc, x_cat


class _BaseTabM(BaseEstimator):
    """Shared implementation for TabMClassifier / TabMRegressor.

    Parameters mirror ``sklearn.neural_network.MLP*`` where sensible;
    TabM-specific options are appended. See the concrete classes for the
    full documentation.
    """

    classes_: np.ndarray
    preprocessor_: _Preprocessor
    core_: TabMCore
    params_: TabMParams
    _space: FlatSpace

    def __init__(
        self,
        hidden_layer_sizes=(256,),
        *,
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=200,
        tol=1e-4,
        verbose=False,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=None,
        activation="relu",
        k=32,
        arch_type="tabm",
        dropout=0.1,
        use_embeddings=True,
        n_bins=64,
        d_embedding=8,
        use_quantile=True,
        use_asinh=True,
        use_scaler=True,
        quantization="none",
        quantization_granularity="per_row",
        categorical_indices=None,
        categorical_cardinality_threshold=32,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.activation = activation
        self.k = k
        self.arch_type = arch_type
        self.dropout = dropout
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.use_quantile = use_quantile
        self.use_asinh = use_asinh
        self.use_scaler = use_scaler
        self.quantization = quantization
        self.quantization_granularity = quantization_granularity
        self.categorical_indices = categorical_indices
        self.categorical_cardinality_threshold = categorical_cardinality_threshold

    # -- helpers -------------------------------------------------------------

    def _hidden_dim(self) -> int:
        sizes = self.hidden_layer_sizes
        if isinstance(sizes, (list, tuple)):
            return int(sizes[0]) if len(sizes) else 256
        return int(sizes)

    def _n_blocks(self) -> int:
        sizes = self.hidden_layer_sizes
        if isinstance(sizes, (list, tuple)):
            return max(1, len(sizes))
        return 1

    def _batch_size(self, n_samples: int) -> int:
        if self.batch_size == "auto":
            return min(200, n_samples)
        return int(self.batch_size)

    def _check_solver(self) -> None:
        if self.solver not in ("adam", "sgd", "lbfgs"):
            raise ValueError(
                f"solver must be 'adam', 'sgd' or 'lbfgs', got {self.solver!r}"
            )
        validate_quantization(self.quantization, self.quantization_granularity)

    def _task_code(self) -> int:
        """Native loss code: 0 regression, 1 binary, 2 multiclass."""
        if hasattr(self, "classes_"):
            return 1 if len(self.classes_) == 2 else 2
        return 0

    def _resolve_backend(self) -> str:
        backend = tabm_backend.get_tabm_backend()
        if backend == "mojo" and self.arch_type != "tabm":
            warnings.warn(
                "The Mojo trainer currently supports arch_type='tabm' only; "
                f"falling back to NumPy for arch_type={self.arch_type!r}."
            )
            return "numpy"
        return backend

    def _build_config(self, pre: _Preprocessor, d_out: int) -> TabMConfig:
        return TabMConfig(
            n_num_features=len(pre.numerical_indices_),
            cat_cardinalities=list(pre.cardinalities_),
            d_out=d_out,
            k=self.k,
            n_blocks=self._n_blocks(),
            d_block=self._hidden_dim(),
            dropout=self.dropout,
            arch_type=self.arch_type,
            use_embeddings=self.use_embeddings,
            bins=list(pre.bins_) if pre.bins_ else None,
            d_embedding=self.d_embedding,
            quantization=self.quantization,
            quantization_granularity=self.quantization_granularity,
        )

    def _split_validation(
        self, X: np.ndarray, y: np.ndarray, stratify: np.ndarray | None
    ):
        if not self.early_stopping:
            return X, y, None, None
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )
        return X_train, y_train, X_val, y_val

    def _make_batch(self, pre: _Preprocessor, X: np.ndarray, y: np.ndarray) -> Batch:
        x_num, x_enc, x_cat = pre.transform(X)
        return Batch(x_num, x_enc, x_cat, np.asarray(y, dtype=np.float32))

    # -- training ------------------------------------------------------------

    def _fit_core(
        self,
        core: TabMCore,
        params: TabMParams,
        train: Batch,
        val: Batch | None,
        seed: int,
    ) -> None:
        """Run the selected solver; populates loss_curve_/n_iter_/etc."""
        self._check_solver()
        backend = self._resolve_backend()
        space = FlatSpace(params)
        theta = params.flatten()
        alpha = float(self.alpha)

        def full_loss_grad(t: np.ndarray) -> tuple[float, np.ndarray]:
            space.scatter(t, params)
            loss, grads = core.loss_and_grads(params, train)
            g = space.flatten_grads(grads)
            if alpha > 0.0:
                g += (alpha * t).astype(np.float32)
                loss += 0.5 * alpha * float(t.astype(np.float64) @ t.astype(np.float64))
            return loss, g.astype(np.float32)

        if self.solver == "lbfgs":
            if self.early_stopping:
                warnings.warn(
                    "early_stopping is not supported with solver='lbfgs'; "
                    "it will be ignored."
                )
            if backend == "mojo":
                from shinrin._tabm._mojo_trainer import get_native_trainer

                n_iter, losses = get_native_trainer(params.config).lbfgs(
                    theta,
                    train,
                    params,
                    space,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    alpha=alpha,
                    task=self._task_code(),
                )
            else:
                theta, n_iter, losses = lbfgs_minimize(
                    full_loss_grad, theta, max_iter=self.max_iter, tol=self.tol
                )
            self.loss_curve_ = losses
            self.n_iter_ = n_iter
            space.scatter(theta, params)
            return

        if self.solver == "sgd" and backend == "mojo":
            backend = "numpy"

        lr = self.learning_rate_init
        state = AdamState(space.total) if self.solver == "adam" else None
        velocity = None
        # Bind the parameter arrays to views of theta so in-place optimizer
        # updates are visible to forward/backward without re-scattering.
        space.scatter(theta, params)
        best_theta = None
        best_score = np.inf
        patience_left = self.n_iter_no_change
        self.loss_curve_ = []
        self.validation_scores_ = []
        n = train.n_samples
        bs = self._batch_size(n)

        for epoch in range(self.max_iter):
            if backend == "mojo":
                from shinrin._tabm._mojo_trainer import get_native_trainer

                assert state is not None
                epoch_loss, state.t = get_native_trainer(params.config).adam_epoch(
                    theta,
                    state.m,
                    state.v,
                    state.t,
                    train,
                    params,
                    space,
                    lr=lr,
                    batch_size=bs,
                    dropout=self.dropout,
                    alpha=alpha,
                    seed=seed + epoch,
                    task=self._task_code(),
                )
            else:
                rng = np.random.RandomState(seed + epoch)
                perm = rng.permutation(n)
                total = 0.0
                for start in range(0, n, bs):
                    sub = train.take(perm[start : start + bs])
                    loss, grads = core.loss_and_grads(params, sub, rng=rng)
                    g = space.flatten_grads(grads)
                    if alpha > 0.0:
                        g += (alpha * theta).astype(np.float32)
                    if state is not None:
                        adam_step(theta, g, state, lr)
                    else:
                        velocity = sgd_step(theta, g, velocity, lr)
                    total += loss * sub.n_samples
                epoch_loss = total / n
            self.loss_curve_.append(epoch_loss)

            if val is not None:
                space.scatter(theta, params)
                val_loss, _ = core.loss_and_grads(params, val)
                self.validation_scores_.append(val_loss)
                if val_loss < best_score - self.tol:
                    best_score = val_loss
                    best_theta = theta.copy()
                    patience_left = self.n_iter_no_change
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if self.verbose and (epoch % 10 == 9 or epoch == self.max_iter - 1):
                msg = f"Iteration {epoch + 1}, loss = {epoch_loss:.6f}"
                if val is not None:
                    msg += f", validation loss = {self.validation_scores_[-1]:.6f}"
                print(msg)

        if best_theta is not None:
            space.scatter(best_theta, params)
        self.best_validation_score_ = best_score if val is not None else None
        self.n_iter_ = len(self.loss_curve_)

    # -- incremental fitting ---------------------------------------------------

    def _partial_fit_epoch(self, X: np.ndarray, y: np.ndarray) -> _BaseTabM:
        if self.solver == "lbfgs":
            raise ValueError("partial_fit requires solver='adam' or 'sgd'")
        assert self.preprocessor_ is not None and self.core_ is not None
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float32)
        batch = self._make_batch(self.preprocessor_, X, y)
        seed = int(self.random_state or 0) + getattr(self, "n_iter_", 0)
        if not hasattr(self, "_adam_state"):
            self._adam_state = AdamState(self._space.total)
        params = self.params_
        theta = params.flatten()
        self._space.scatter(theta, params)
        rng = np.random.RandomState(seed)
        n = batch.n_samples
        bs = self._batch_size(n)
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, bs):
            sub = batch.take(perm[start : start + bs])
            loss, grads = self.core_.loss_and_grads(params, sub, rng=rng)
            g = self._space.flatten_grads(grads)
            if self.alpha > 0.0:
                g += (self.alpha * theta).astype(np.float32)
            if self.solver == "adam":
                adam_step(theta, g, self._adam_state, self.learning_rate_init)
            else:
                self._sgd_velocity = sgd_step(
                    theta,
                    g,
                    getattr(self, "_sgd_velocity", None),
                    self.learning_rate_init,
                )
            total += loss * sub.n_samples
        self._space.scatter(theta, params)
        self.loss_curve_.append(total / n)
        self.n_iter_ += 1
        return self


class TabMClassifier(ClassifierMixin, _BaseTabM):
    """TabM classifier with a scikit-learn compatible interface."""

    """TabM classifier with a scikit-learn compatible interface.

    A drop-in replacement for ``sklearn.neural_network.MLPClassifier``
    backed by TabM — one model efficiently imitating an ensemble of ``k``
    MLPs via parameter-efficient ensembling (ICLR 2025).

    Training uses pure NumPy (optionally Mojo-accelerated via
    ``SHINRIN_TABM_BACKEND=mojo``); PyTorch is not required.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(256,)
        Width of hidden layers; the tuple length sets the number of
        residual MLP blocks (all layers share the same width).
    solver : {'adam', 'sgd', 'lbfgs'}, default='adam'
        'adam'/'sgd' run minibatch epochs; 'lbfgs' runs full-batch
        L-BFGS (dropout disabled, deterministic).
    alpha : float, default=1e-4
        L2 penalty strength added to the gradient of all parameters.
    batch_size : int or 'auto', default='auto'
        Minibatch size ('auto' -> ``min(200, n_samples)``).
    learning_rate_init : float, default=1e-3
        Initial step size for adam/sgd.
    max_iter : int, default=200
        Maximum number of epochs (L-BFGS iterations for 'lbfgs').
    tol : float, default=1e-4
        Early-stopping tolerance / L-BFGS gradient tolerance.
    verbose : bool, default=False
    early_stopping : bool, default=False
        Reserve a validation split and restore the best weights.
    validation_fraction : float, default=0.1
    n_iter_no_change : int, default=10
        Patience (epochs) used with ``early_stopping``.
    random_state : int or None, default=None
    activation : {'relu'}, default='relu'
        TabM uses ReLU internally; other values are rejected.
    k : int, default=32
        Ensemble size (number of implicit MLP members).
    arch_type : {'tabm', 'tabm-mini', 'tabm-packed'}, default='tabm'
    dropout : float, default=0.1
    use_embeddings : bool, default=True
        Piecewise-linear embeddings for numerical features.
    n_bins : int, default=64
        Bins per feature for the piecewise-linear embedding.
    d_embedding : int, default=8
        Embedding dimension per numerical feature.
    use_quantile, use_asinh, use_scaler : bool, default=True
        Preprocessing transforms applied to numerical features.
    quantization : {'none', 'ternary'}, default='none'
        BitNet-style training-aware ternary weight quantization of the
        shared backbone blocks (BitLinear). Latent float32 weights are
        trained with straight-through gradients while the forward pass
        uses the ``{-1, 0, +1} * gamma`` approximation; embeddings,
        adapters, biases and the head stay at full precision.
    quantization_granularity : {'per_row', 'per_tensor'}, default='per_row'
        Scale granularity for the ternary approximation.
    categorical_indices : list of int, default=None
        Columns forced categorical (others auto-detected by cardinality).
    categorical_cardinality_threshold : int, default=32
        Max unique values for auto-detection (0 disables).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    loss_curve_ : list of float
    validation_scores_ : list of float
        Per-epoch validation loss when ``early_stopping=True``.
    n_iter_ : int
    n_features_in_ : int

    Examples
    --------
    >>> from shinrin import TabMClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = TabMClassifier(hidden_layer_sizes=(128,), max_iter=20).fit(X, y)
    >>> clf.score(X, y) > 0.9
    True
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def _setup_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        known_classes: np.ndarray | None = None,
        train_now: bool = True,
    ) -> None:
        self._check_solver()
        if self.activation != "relu":
            raise ValueError("TabMClassifier only supports activation='relu'")
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = (
            np.asarray(known_classes) if known_classes is not None else unique_labels(y)
        )
        self._label_encoder = LabelEncoder().fit(self.classes_)
        y_enc = self._label_encoder.transform(y).astype(np.float32)
        self.n_outputs_ = 1
        self.out_activation_ = "logistic" if len(self.classes_) == 2 else "softmax"
        task = "binary" if len(self.classes_) == 2 else "multiclass"
        d_out = 1 if task == "binary" else len(self.classes_)

        X_train, y_train, X_val, y_val = self._split_validation(
            X, y_enc, stratify=y_enc
        )
        self.preprocessor_ = _Preprocessor(self).fit(X_train)
        self.config_ = self._build_config(self.preprocessor_, d_out)
        self.core_ = TabMCore(self.config_, task)
        self.params_ = TabMParams.init(self.config_, seed=_seed(self.random_state))
        self._space = FlatSpace(self.params_)

        if not train_now:
            self.loss_curve_ = []
            self.n_iter_ = 0
            return
        train = self._make_batch(self.preprocessor_, X_train, y_train)
        val_batch = (
            self._make_batch(self.preprocessor_, X_val, y_val)
            if X_val is not None
            else None
        )
        self._fit_core(
            self.core_,
            self.params_,
            train,
            val_batch,
            seed=_seed(self.random_state),
        )

    def fit(self, X, y):
        """Fit the model to training data ``X`` and labels ``y``."""
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False, y_numeric=False)
        self._setup_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Incrementally fit one epoch on a batch of samples."""
        first = not hasattr(self, "params_")
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        if first:
            if classes is None:
                raise ValueError(
                    "classes must be provided on the first call to partial_fit"
                )
            self._setup_fit(X, y, known_classes=classes, train_now=False)
        return self._partial_fit_epoch(X, y)

    def _decision(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "params_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float32)
        batch = self._make_batch(self.preprocessor_, X, np.zeros(len(X)))
        return self.core_.predict(self.params_, batch)  # (B, d_out)

    def predict_proba(self, X):
        """Probability estimates; returns ``(n_samples, n_classes)``."""
        decision = self._decision(np.asarray(X))
        if self.out_activation_ == "logistic":
            p = 1.0 / (1.0 + np.exp(-decision[:, 0]))
            return np.column_stack([1.0 - p, p])
        shifted = decision - decision.max(axis=1, keepdims=True)
        proba = np.exp(shifted)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict the most likely class for each sample."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(proba.argmax(axis=1))

    def score(self, X, y, sample_weight=None):
        """Mean accuracy on the given data and labels."""
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class TabMRegressor(RegressorMixin, _BaseTabM):
    """TabM regressor with a scikit-learn compatible interface.

    A drop-in replacement for ``sklearn.neural_network.MLPRegressor``
    backed by TabM (ICLR 2025). See :class:`TabMClassifier` for the full
    parameter documentation; behaviour is identical apart from the target
    type (continuous, possibly multi-output).

    Attributes
    ----------
    loss_curve_ : list of float
    n_iter_ : int
    n_features_in_ : int
    n_outputs_ : int

    Examples
    --------
    >>> from shinrin import TabMRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = TabMRegressor(hidden_layer_sizes=(128,), max_iter=50).fit(X, y)
    >>> reg.score(X, y) > 0.2
    True
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _setup_fit(self, X: np.ndarray, y: np.ndarray, train_now: bool = True) -> None:
        self._check_solver()
        if self.activation != "relu":
            raise ValueError("TabMRegressor only supports activation='relu'")
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_outputs = 1 if y.ndim == 1 else y.shape[1]
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was "
                "expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
            )
            y = y.ravel()
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = n_outputs
        self.out_activation_ = "identity"

        X_train, y_train, X_val, y_val = self._split_validation(X, y, stratify=None)
        self.preprocessor_ = _Preprocessor(self).fit(X_train)
        self.config_ = self._build_config(self.preprocessor_, self.n_outputs_)
        self.core_ = TabMCore(self.config_, "regression")
        self.params_ = TabMParams.init(self.config_, seed=_seed(self.random_state))
        self._space = FlatSpace(self.params_)

        if not train_now:
            self.loss_curve_ = []
            self.n_iter_ = 0
            return
        train = self._make_batch(self.preprocessor_, X_train, y_train)
        val_batch = (
            self._make_batch(self.preprocessor_, X_val, y_val)
            if X_val is not None
            else None
        )
        self._fit_core(
            self.core_,
            self.params_,
            train,
            val_batch,
            seed=_seed(self.random_state),
        )

    def fit(self, X, y):
        """Fit the model to training data ``X`` and targets ``y``."""
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True, y_numeric=True)
        self._setup_fit(np.asarray(X), np.asarray(y))
        return self

    def partial_fit(self, X, y):
        """Incrementally fit one epoch on a batch of samples."""
        first = not hasattr(self, "params_")
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True, y_numeric=True)
        if first:
            self._setup_fit(X, y, train_now=False)
            return self._partial_fit_epoch(X, y)
        return self._partial_fit_epoch(X, y)

    def predict(self, X):
        """Predict regression targets; squeezes single-output predictions."""
        check_is_fitted(self, "params_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float32)
        batch = self._make_batch(self.preprocessor_, X, np.zeros((len(X), 1)))
        out = self.core_.predict(self.params_, batch)
        if self.n_outputs_ == 1:
            return out[:, 0]
        return out

    def score(self, X, y, sample_weight=None):
        """R^2 score on the given data."""
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X), sample_weight=sample_weight)


def _seed(random_state: Any) -> int:
    if random_state is None:
        return np.random.RandomState().randint(0, 2**31 - 1)
    if isinstance(random_state, (int, np.integer)):
        return int(random_state) % (2**31 - 1)
    return int(random_state.randint(0, 2**31 - 1))
