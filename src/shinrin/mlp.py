"""Scikit-learn compatible MLPClassifier / MLPRegressor.

Drop-in replacements for ``sklearn.neural_network.MLPClassifier`` /
``MLPRegressor`` with the same parameters, attributes and training
semantics, backed by the NumPy core in :mod:`shinrin._mlp` and optionally
accelerated by bundled Mojo kernels (``SHINRIN_MLP_BACKEND=mojo``).

Shinrin extensions beyond scikit-learn:

- ``use_embeddings=True`` routes numerical features through piecewise
  linear (PLE) embeddings — quantile bins, a trainable per-feature
  projection and ReLU, mirroring the TabM embedding recipe
- ``dropout`` adds dropout after every hidden layer
- categorical feature detection (``categorical_indices``,
  ``categorical_cardinality_threshold``) feeds one-hot blocks to the
  network
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data

from shinrin._mlp._layers import ACTIVATIONS, MLPConfig, MLPParams
from shinrin._mlp._model import Batch, MLPCore
from shinrin._tabm._optim import AdamState, FlatSpace, lbfgs_minimize
from shinrin._tabm._transforms import (
    AsinhTransform,
    PiecewiseLinearEncoder,
    QuantileTransform,
    StandardScalerTransform,
    build_num_bins,
    detect_categorical_features,
)

__all__ = ["MLPClassifier", "MLPRegressor"]


class _Preprocessor:
    """Fitted preprocessing pipeline (transforms + PLE encoding)."""

    def __init__(self, base: _BaseMLP) -> None:
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
            current = X[:, num_idx]
            if p.use_quantile:
                t = QuantileTransform(num_quantiles=100).fit(current)
                current = t.transform(current)
                self.transforms_.append(t)
            if p.use_asinh:
                t = AsinhTransform().fit(current)
                current = t.transform(current)
                self.transforms_.append(t)
            if p.use_scaler or (p.use_embeddings and not self.transforms_):
                # Standardization is part of the default PLE embedding
                # recipe; standalone it is opt-in via use_scaler.
                t = StandardScalerTransform().fit(current)
                current = t.transform(current)
                self.transforms_.append(t)
            if p.use_embeddings:
                self.bins_ = build_num_bins(current, p.n_bins)
                self.encoder_ = PiecewiseLinearEncoder(self.bins_)
            else:
                self.bins_ = []
                self.encoder_ = None
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

    def d_in_effective(self) -> int:
        """Width of the flat vector fed to the first weight matrix."""
        n_num = len(self.numerical_indices_)
        if self.encoder_ is not None and self.base.use_embeddings and n_num:
            num = n_num * int(self.base.d_embedding)
        else:
            num = n_num
        return num + sum(self.cardinalities_)

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


class _SGDOptimizer:
    """SGD with momentum / nesterov matching sklearn's formulas."""

    def __init__(
        self,
        total: int,
        lr_init: float,
        lr_schedule: str,
        power_t: float,
        momentum: float,
        nesterov: bool,
    ) -> None:
        self.lr = float(lr_init)
        self.lr_init = float(lr_init)
        self.lr_schedule = lr_schedule
        self.power_t = float(power_t)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.velocity = np.zeros(total, dtype=np.float32) if momentum > 0.0 else None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> None:
        """One minibatch update on ``theta`` (in place).

        Mirrors sklearn's ``SGDOptimizer``: velocities carry the descent
        direction (``v = mu * v - lr * grad``) and parameters receive the
        velocity directly (``theta += v``).
        """
        if self.velocity is None:
            theta -= (self.lr * grad).astype(np.float32)
            return
        vel = self.velocity
        vel *= np.float32(self.momentum)
        vel -= np.float32(self.lr) * grad
        update = vel
        if self.nesterov:
            update = np.float32(self.momentum) * vel - np.float32(self.lr) * grad
        theta += update.astype(np.float32)

    def iteration_ends(self, t_: float) -> None:
        if self.lr_schedule == "invscaling":
            self.lr = self.lr_init / float(t_ + 1.0) ** self.power_t

    def trigger_stopping(self, msg: str, verbose: bool) -> bool:
        """Return True when training should stop (sklearn semantics)."""
        if self.lr_schedule != "adaptive":
            if verbose:
                print(msg + " Stopping.")
            return True
        if self.lr <= 1e-6:
            if verbose:
                print(msg + " Learning rate too small. Stopping.")
            return True
        self.lr /= 5.0
        if verbose:
            print(msg + f" Setting learning rate to {self.lr:f}")
        return False


class _AdamOptimizer(AdamState):
    """Adam honouring sklearn's beta_1 / beta_2 / epsilon parameters."""

    def __init__(
        self, total: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ) -> None:
        super().__init__(total)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def step(self, theta: np.ndarray, grad: np.ndarray, lr: float) -> None:
        self.t += 1
        b1, b2 = np.float32(self.beta1), np.float32(self.beta2)
        self.m = b1 * self.m + (1.0 - b1) * grad
        self.v = b2 * self.v + (1.0 - b2) * (grad * grad)
        m_hat = self.m / np.float32(1.0 - self.beta1**self.t)
        v_hat = self.v / np.float32(1.0 - self.beta2**self.t)
        theta -= (lr * m_hat / (np.sqrt(v_hat) + np.float32(self.eps))).astype(
            np.float32
        )

    def iteration_ends(self, t_: float) -> None:
        """No-op; Adam keeps a constant learning rate (sklearn semantics)."""

    def trigger_stopping(self, msg: str, verbose: bool) -> bool:
        if verbose:
            print(msg + " Stopping.")
        return True


class _BaseMLP(BaseEstimator):
    """Shared implementation for MLPClassifier / MLPRegressor.

    Parameters mirror ``sklearn.neural_network.MLP*``; shinrin-specific
    options are appended. See the concrete classes for full docs.
    """

    classes_: np.ndarray
    preprocessor_: _Preprocessor
    core_: MLPCore
    params_: MLPParams
    config_: MLPConfig
    out_activation_: str
    n_outputs_: int
    coefs_: list[np.ndarray]
    intercepts_: list[np.ndarray]
    loss_curve_: list[float]
    validation_scores_: list[float] | None
    best_validation_score_: float | None
    best_loss_: float | None
    t_: float
    n_iter_: int
    _space: FlatSpace

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        *,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=1e-3,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        dropout=0.0,
        use_embeddings=False,
        n_bins=64,
        d_embedding=8,
        use_quantile=False,
        use_asinh=False,
        use_scaler=False,
        categorical_indices=None,
        categorical_cardinality_threshold=32,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        # -- shinrin extensions -------------------------------------------------
        self.dropout = dropout
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.use_quantile = use_quantile
        self.use_asinh = use_asinh
        self.use_scaler = use_scaler
        self.categorical_indices = categorical_indices
        self.categorical_cardinality_threshold = categorical_cardinality_threshold

    # -- validation helpers ---------------------------------------------------

    def _check_params(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {self.activation!r}"
            )
        if self.solver not in ("adam", "sgd", "lbfgs"):
            raise ValueError(
                f"solver must be 'adam', 'sgd' or 'lbfgs', got {self.solver!r}"
            )
        if self.learning_rate not in ("constant", "invscaling", "adaptive"):
            raise ValueError(
                "learning_rate must be 'constant', 'invscaling' or "
                f"'adaptive', got {self.learning_rate!r}"
            )
        # NOTE: like sklearn, non-'constant' rates are silently ignored by
        # the Adam solver (its iteration_ends is a no-op).
        if isinstance(self.batch_size, str) and self.batch_size != "auto":
            raise ValueError("batch_size must be an int or 'auto'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def _hidden_dims(self) -> list[int]:
        sizes = self.hidden_layer_sizes
        if isinstance(sizes, (list, tuple)):
            dims = [int(s) for s in sizes]
            return dims or [100]
        return [int(sizes)]

    def _batch_size(self, n_samples: int) -> int:
        if isinstance(self.batch_size, str):
            return min(200, n_samples)
        return max(1, min(int(self.batch_size), n_samples))

    def _resolve_backend(self) -> str:
        from shinrin._mlp._backend import get_mlp_backend

        backend = get_mlp_backend()
        if backend == "mojo" and self.solver != "adam":
            warnings.warn(
                "The Mojo trainer supports solver='adam' only; falling back "
                f"to NumPy for solver={self.solver!r}."
            )
            return "numpy"
        return backend

    def _task_code(self) -> int:
        """Native loss code: 0 regression, 1 binary, 2 multiclass."""
        return {"identity": 0, "logistic": 1, "softmax": 2}[self.out_activation_]

    def _build_config(self, pre: _Preprocessor, d_out: int) -> MLPConfig:
        layer_sizes = [pre.d_in_effective()] + self._hidden_dims() + [d_out]
        return MLPConfig(
            n_num_features=len(pre.numerical_indices_),
            cat_cardinalities=list(pre.cardinalities_),
            d_out=d_out,
            layer_sizes=layer_sizes,
            activation=self.activation,
            dropout=float(self.dropout),
            use_embeddings=self.use_embeddings,
            bins=list(pre.bins_) if pre.bins_ else None,
            d_embedding=int(self.d_embedding),
        )

    def _split_validation(self, X, y, stratify=None):
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

    # -- training -----------------------------------------------------------------

    def _fit_core(
        self,
        core: MLPCore,
        params: MLPParams,
        train: Batch,
        val: Batch | None,
        val_score_fn,
        seed: int,
    ) -> None:
        """Run the selected solver; populates loss_curve_/n_iter_/coefs_ etc."""
        backend = self._resolve_backend()
        space = FlatSpace(params)
        self._space = space
        theta = params.flatten()
        space.scatter(theta, params)
        alpha = float(self.alpha)
        task_code = self._task_code()

        def full_loss_grad(t: np.ndarray) -> tuple[float, np.ndarray]:
            space.scatter(t, params)
            loss, grads = core.loss_and_grads(params, train)
            g = space.flatten_grads(grads)
            if alpha > 0.0:
                g += (alpha * t).astype(np.float32)
                loss += 0.5 * alpha * float(t.astype(np.float64) @ t.astype(np.float64))
            return loss, g.astype(np.float32)

        if self.solver == "lbfgs":
            self._fit_lbfgs(full_loss_grad, theta, space, params, train)
            return

        # ---- stochastic solvers (adam / sgd) ------------------------------------
        n = train.n_samples
        bs = self._batch_size(n)
        if self.solver == "adam":
            opt: Any = _AdamOptimizer(
                space.total, self.beta_1, self.beta_2, self.epsilon
            )
        else:
            opt = _SGDOptimizer(
                space.total,
                self.learning_rate_init,
                self.learning_rate,
                self.power_t,
                self.momentum,
                bool(self.nesterovs_momentum),
            )
        best_theta = None
        no_improvement = 0
        self.loss_curve_ = []
        self.validation_scores_ = []
        self.best_validation_score_ = -np.inf
        best_loss = float(self.best_loss_) if self.best_loss_ is not None else np.inf
        base_seed = seed

        for epoch in range(self.max_iter):
            epoch_rng = np.random.RandomState(base_seed + epoch)
            perm = epoch_rng.permutation(n) if self.shuffle else np.arange(n)
            if backend == "mojo":
                assert isinstance(opt, _AdamOptimizer)
                epoch_loss, opt.t = self._mojo_epoch(
                    theta, opt, train, bs, alpha, base_seed + epoch, task_code
                )
            else:
                epoch_loss = self._stochastic_epoch(
                    core, params, space, theta, opt, train, perm, bs, alpha, epoch_rng
                )
            self.loss_curve_.append(epoch_loss)
            self.t_ += float(n)

            # -- update no-improvement count (mirrors sklearn exactly) ----------
            if self.early_stopping and val is not None:
                space.scatter(theta, params)
                score = float(val_score_fn(core, params, val))
                self.validation_scores_.append(score)
                if score < self.best_validation_score_ + self.tol:
                    no_improvement += 1
                else:
                    no_improvement = 0
                if score > self.best_validation_score_:
                    self.best_validation_score_ = score
                    best_theta = theta.copy()
            else:
                if self.loss_curve_[-1] > best_loss - self.tol:
                    no_improvement += 1
                else:
                    no_improvement = 0
                best_loss = min(best_loss, self.loss_curve_[-1])

            opt.iteration_ends(self.t_)

            if no_improvement > self.n_iter_no_change:
                msg = (
                    "Validation score did not improve more than "
                    f"tol={self.tol:f} for {self.n_iter_no_change} consecutive epochs."
                    if self.early_stopping
                    else (
                        "Training loss did not improve more than "
                        f"tol={self.tol:f} for {self.n_iter_no_change} consecutive epochs."
                    )
                )
                stopping = opt.trigger_stopping(msg, self.verbose)
                if stopping:
                    break
                no_improvement = 0

            if self.verbose and (epoch % 10 == 9 or epoch == self.max_iter - 1):
                msg = f"Iteration {epoch + 1}, loss = {epoch_loss:.8f}"
                print(msg)

        if self.early_stopping and best_theta is not None:
            # restore best weights (sklearn restores unconditionally when set)
            space.scatter(best_theta, params)
        else:
            self.best_loss_ = best_loss
        self.n_iter_ = len(self.loss_curve_)
        self._publish_coefs(params)

    def _fit_lbfgs(
        self,
        full_loss_grad,
        theta: np.ndarray,
        space: FlatSpace,
        params: MLPParams,
        train: Batch,
    ) -> None:
        if self.early_stopping:
            warnings.warn(
                "early_stopping is not supported with solver='lbfgs'; it will be ignored.",
                UserWarning,
            )
        theta, nit, losses = lbfgs_minimize(
            full_loss_grad,
            theta,
            max_iter=min(self.max_iter, int(self.max_fun)),
            tol=self.tol,
        )
        space.scatter(theta, params)
        self.loss_curve_ = losses
        self.best_loss_ = float(min(losses))
        self.n_iter_ = nit
        self.t_ = float(getattr(self, "t_", 0.0)) + float(train.n_samples * len(losses))
        self._publish_coefs(params)

    def _stochastic_epoch(
        self,
        core: MLPCore,
        params: MLPParams,
        space: FlatSpace,
        theta: np.ndarray,
        opt: Any,
        train: Batch,
        perm: np.ndarray,
        bs: int,
        alpha: float,
        rng: np.random.RandomState,
    ) -> float:
        n = train.n_samples
        total = 0.0
        for start in range(0, n, bs):
            sub = train.take(perm[start : start + bs])
            loss, grads = core.loss_and_grads(params, sub, rng=rng)
            g = space.flatten_grads(grads)
            if alpha > 0.0:
                g += (alpha * theta).astype(np.float32)
            if isinstance(opt, _AdamOptimizer):
                opt.step(theta, g, self.learning_rate_init)
            else:
                opt.step(theta, g)
            total += loss * sub.n_samples
        return total / n

    def _mojo_epoch(
        self,
        theta: np.ndarray,
        opt: Any,
        train: Batch,
        bs: int,
        alpha: float,
        seed: int,
        task_code: int,
    ) -> tuple[float, int]:
        from shinrin._mlp._mojo_trainer import get_native_trainer

        native = get_native_trainer(self.config_)
        return native.adam_epoch(
            theta,
            opt.m,
            opt.v,
            opt.t,
            train,
            self.config_,
            self._space,
            lr=self.learning_rate_init,
            batch_size=bs,
            dropout=float(self.dropout),
            alpha=alpha,
            seed=seed,
            task=task_code,
        )

    def _publish_coefs(self, params: MLPParams) -> None:
        self.coefs_ = params.coefs_()
        self.intercepts_ = params.intercepts_()
        self.n_layers_ = self.config_.n_layers + 1  # counting the input layer

    def _partial_fit_epoch(self, X: np.ndarray, y: np.ndarray) -> None:
        """One minibatch-epoch of incremental fitting (NumPy backend)."""
        assert self.core_ is not None and self.params_ is not None
        if not hasattr(self, "_space") or not hasattr(self, "_opt_state"):
            self._space = FlatSpace(self.params_)
            if self.solver == "adam":
                self._opt_state: Any = _AdamOptimizer(
                    self._space.total, self.beta_1, self.beta_2, self.epsilon
                )
            else:
                self._opt_state = _SGDOptimizer(
                    self._space.total,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.power_t,
                    self.momentum,
                    bool(self.nesterovs_momentum),
                )
            self.best_loss_ = np.inf
            self.no_improvement_ = 0
        params = self.params_
        space = self._space
        theta = params.flatten()
        space.scatter(theta, params)
        batch = self._make_batch(self.preprocessor_, X, y)
        rng = np.random.RandomState(
            _seed(self.random_state) + getattr(self, "n_iter_", 0)
        )
        n = batch.n_samples
        bs = self._batch_size(n)
        perm = rng.permutation(n) if self.shuffle else np.arange(n)
        total = 0.0
        for start in range(0, n, bs):
            sub = batch.take(perm[start : start + bs])
            loss, grads = self.core_.loss_and_grads(params, sub, rng=rng)
            g = space.flatten_grads(grads)
            if self.alpha > 0.0:
                g += (self.alpha * theta).astype(np.float32)
            if isinstance(self._opt_state, _AdamOptimizer):
                self._opt_state.step(theta, g, self.learning_rate_init)
            else:
                self._opt_state.step(theta, g)
            total += loss * sub.n_samples
        space.scatter(theta, params)
        self.loss_curve_.append(total / n)
        self.best_loss_ = min(float(self.best_loss_ or np.inf), self.loss_curve_[-1])
        self.n_iter_ = getattr(self, "n_iter_", 0) + 1
        self.t_ = getattr(self, "t_", 0.0) + float(n)
        self._opt_state.iteration_ends(self.t_)
        self._publish_coefs(params)


def _seed(random_state: Any) -> int:
    if random_state is None:
        return int(np.random.RandomState().randint(0, 2**31 - 1))
    if isinstance(random_state, (int, np.integer)):
        return int(random_state) % (2**31 - 1)
    return int(random_state.randint(0, 2**31 - 1))


class MLPClassifier(ClassifierMixin, _BaseMLP):
    """MLP classifier matching ``sklearn.neural_network.MLPClassifier``.

    A drop-in replacement with the same parameter surface
    (``hidden_layer_sizes``, ``activation``, ``solver``, ``alpha``,
    ``batch_size``, ``learning_rate``, ``momentum``, ``early_stopping``,
    ...) and attributes (``classes_``, ``coefs_``, ``intercepts_``,
    ``loss_curve_``, ``validation_scores_``, ``n_iter_``), trained by the
    shinrin NumPy/Mojo backend.

    Shinrin extensions:

    - ``use_embeddings=True``: piecewise-linear embeddings for numerical
      features — quantile bins followed by a trainable per-feature linear
      projection with ReLU (the TabM embedding recipe). Recommended
      together with ``use_asinh=True, use_scaler=True``.
    - ``dropout``: dropout probability applied after every hidden layer.
    - automatic categorical detection (cardinality threshold) with
      one-hot encoding; force columns via ``categorical_indices``.

    Examples
    --------
    >>> from shinrin import MLPClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=20).fit(X, y)
    >>> clf.score(X, y) > 0.9
    True
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y):
        """Fit the model to training data ``X`` and labels ``y``."""
        self._check_params()
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False, y_numeric=False)
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)

        warm = self.warm_start and hasattr(self, "coefs_")
        if not warm:
            self.n_features_in_ = X.shape[1]
            self.classes_ = unique_labels(y)
            self._label_encoder = LabelEncoder().fit(self.classes_)
            self.n_outputs_ = 1
            self.out_activation_ = "logistic" if len(self.classes_) == 2 else "softmax"
            task = "binary" if len(self.classes_) == 2 else "multiclass"
            d_out = 1 if task == "binary" else len(self.classes_)
            y_enc = self._label_encoder.transform(y).astype(np.float32)
            X_train, y_train, X_val, y_val = self._split_validation(
                X, y_enc, stratify=y_enc
            )
            self.preprocessor_ = _Preprocessor(self).fit(X_train)
            self.config_ = self._build_config(self.preprocessor_, d_out)
            self.core_ = MLPCore(self.config_, task)
            seed = _seed(self.random_state)
            self.params_ = MLPParams.init(self.config_, seed=seed)
            self.t_ = 0.0
            # sklearn keeps best_loss_ as None while early stopping tracks
            # the validation score instead of the training loss.
            self.best_loss_ = None if self.early_stopping else np.inf
        else:
            y_enc = self._label_encoder.transform(y).astype(np.float32)
            X_train, y_train, X_val, y_val = self._split_validation(
                X, y_enc, stratify=y_enc
            )
            seed = _seed(self.random_state)

        def val_score_fn(core, params, val_batch):
            preds = core.predict(params, val_batch)
            true = val_batch.y[:, 0].astype(np.int64)
            if self.out_activation_ == "logistic":
                pred_labels = (preds[:, 0] >= 0.0).astype(np.int64)
            else:
                pred_labels = preds.argmax(axis=1)
            return float((pred_labels == true).mean())

        train = self._make_batch(self.preprocessor_, X_train, y_train)
        val_batch = (
            self._make_batch(self.preprocessor_, X_val, y_val)
            if X_val is not None
            else None
        )
        self._fit_core(self.core_, self.params_, train, val_batch, val_score_fn, seed)
        if self.solver != "lbfgs" and self.n_iter_ >= self.max_iter:
            warnings.warn(
                f"Stochastic Optimizer: Maximum iterations ({self.max_iter}) "
                "reached and the optimization hasn't converged yet.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def partial_fit(self, X, y, classes=None):
        """Incrementally fit one epoch on a batch of samples."""
        if self.early_stopping:
            raise ValueError("partial_fit does not support early_stopping=True")
        first = not hasattr(self, "params_")
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        if first:
            if classes is None:
                raise ValueError(
                    "classes must be provided on the first call to partial_fit"
                )
            self._check_params()
            X = np.ascontiguousarray(X, dtype=np.float32)
            self.n_features_in_ = X.shape[1]
            self.classes_ = np.asarray(classes)
            self._label_encoder = LabelEncoder().fit(self.classes_)
            self.n_outputs_ = 1
            self.out_activation_ = "logistic" if len(self.classes_) == 2 else "softmax"
            task = "binary" if len(self.classes_) == 2 else "multiclass"
            d_out = 1 if task == "binary" else len(self.classes_)
            y_enc = self._label_encoder.transform(y).astype(np.float32)
            self.preprocessor_ = _Preprocessor(self).fit(X)
            self.config_ = self._build_config(self.preprocessor_, d_out)
            self.core_ = MLPCore(self.config_, task)
            self.params_ = MLPParams.init(self.config_, seed=_seed(self.random_state))
            self.loss_curve_ = []
            self.t_ = 0.0
        else:
            y_enc = self._label_encoder.transform(y).astype(np.float32)
        self._partial_fit_epoch(np.ascontiguousarray(X, dtype=np.float32), y_enc)
        return self

    def predict_proba(self, X):
        """Probability estimates ``(n_samples, n_classes)``."""
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

    def _decision(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coefs_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float32)
        batch = self._make_batch(self.preprocessor_, X, np.zeros(len(X)))
        return self.core_.predict(self.params_, batch)


class MLPRegressor(RegressorMixin, _BaseMLP):
    """MLP regressor matching ``sklearn.neural_network.MLPRegressor``.

    Same parameters and attributes as scikit-learn's regressor; see
    :class:`MLPClassifier` for the shinrin extensions (PLE embeddings,
    dropout, categorical detection).

    Examples
    --------
    >>> from shinrin import MLPRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = MLPRegressor(hidden_layer_sizes=(32,), max_iter=50).fit(X, y)
    >>> reg.score(X, y) > 0.2
    True
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def fit(self, X, y):
        """Fit the model to training data ``X`` and targets ``y``."""
        self._check_params()
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True, y_numeric=True)
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example "
                "using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()

        warm = self.warm_start and hasattr(self, "coefs_")
        if not warm:
            self.n_features_in_ = X.shape[1]
            self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
            self.out_activation_ = "identity"
            X_train, y_train, X_val, y_val = self._split_validation(X, y)
            self.preprocessor_ = _Preprocessor(self).fit(X_train)
            self.config_ = self._build_config(self.preprocessor_, self.n_outputs_)
            self.core_ = MLPCore(self.config_, "regression")
            seed = _seed(self.random_state)
            self.params_ = MLPParams.init(self.config_, seed=seed)
            self.t_ = 0.0
            self.best_loss_ = np.inf
        else:
            X_train, y_train, X_val, y_val = self._split_validation(X, y)
            seed = _seed(self.random_state)

        def val_score_fn(core, params, val_batch):
            preds = core.predict(params, val_batch)
            y_true = val_batch.y
            ss_res = float(np.sum((y_true - preds) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        train = self._make_batch(self.preprocessor_, X_train, y_train)
        val_batch = (
            self._make_batch(self.preprocessor_, X_val, y_val)
            if X_val is not None
            else None
        )
        self._fit_core(self.core_, self.params_, train, val_batch, val_score_fn, seed)
        if self.solver != "lbfgs" and self.n_iter_ >= self.max_iter:
            warnings.warn(
                f"Stochastic Optimizer: Maximum iterations ({self.max_iter}) "
                "reached and the optimization hasn't converged yet.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def partial_fit(self, X, y):
        """Incrementally fit one epoch on a batch of samples."""
        if self.early_stopping:
            raise ValueError("partial_fit does not support early_stopping=True")
        first = not hasattr(self, "params_")
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True, y_numeric=True)
        if first:
            self._check_params()
            X = np.ascontiguousarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            self.n_features_in_ = X.shape[1]
            self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
            self.out_activation_ = "identity"
            self.preprocessor_ = _Preprocessor(self).fit(X)
            self.config_ = self._build_config(self.preprocessor_, self.n_outputs_)
            self.core_ = MLPCore(self.config_, "regression")
            self.params_ = MLPParams.init(self.config_, seed=_seed(self.random_state))
            self.loss_curve_ = []
            self.t_ = 0.0
        self._partial_fit_epoch(
            np.ascontiguousarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        )
        return self

    def predict(self, X):
        """Predict regression targets; squeezes single-output predictions."""
        check_is_fitted(self, "coefs_")
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
