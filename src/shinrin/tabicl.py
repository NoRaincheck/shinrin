"""TabICL: tabular in-context learning estimators (TabICLv2).

Sklearn-compatible classifiers/regressors backed by pre-trained TabICLv2
checkpoints. Weights are fetched once from the ``jingang/TabICL`` Hugging Face
repository, converted to ``.npz`` archives (see :mod:`shinrin._tabicl
._checkpoint`) and shared by every backend (torch, NumPy, Mojo).

Requires the optional dependency group ``tabicl`` (``torch``) unless another
backend is selected via the ``backend`` parameter or the
``SHINRIN_TABICL_BACKEND`` environment variable.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from ._tabicl._backend import get_tabicl_backend
from ._tabicl._checkpoint import ensure_npz
from ._tabicl._config import TabICLConfig
from ._tabicl._preprocess import EnsembleGenerator, TransformToNumerical
from ._tabicl._quantile_dist import QuantileToDistribution
from ._quant import (
    QUANTIZATION_TERNARY,
    validate_quantization,
    ternary_quantize_dequantize,
)

__all__ = ["TabICLClassifier", "TabICLRegressor"]

CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _softmax(
    logits: np.ndarray, axis: int = -1, temperature: float = 1.0
) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _ternary_post_training_quantize(
    params: dict[str, np.ndarray], granularity: str
) -> dict[str, np.ndarray]:
    """Experimental BitLinear PTQ of a TabICL state dict.

    Every 2-D ``*.weight`` tensor except the fused attention QKV
    projections (``*in_proj_weight``) is replaced by its absmean ternary
    approximation: MLP linears and attention output projections degrade
    gracefully while naively ternarizing Q/K/V destroys accuracy
    (~chance level), so those stay full precision. Biases, norms, class
    tokens and rotary tables stay full precision too. Applied once at
    load time so all backends share identical weights.
    """
    out: dict[str, np.ndarray] = {}
    for key, arr in params.items():
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and key.endswith("weight")
            and "in_proj_weight" not in key
        ):
            out[key] = ternary_quantize_dequantize(
                np.asarray(arr, dtype=np.float32), granularity
            ).astype(arr.dtype, copy=False)
        else:
            out[key] = arr
    return out


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Join test-row chunks along the test axis (last for 2-D, axis 1 for
    batched 3-D model outputs)."""
    if len(chunks) == 1:
        return chunks[0]
    axis = 1 if chunks[0].ndim == 3 else 0
    return np.concatenate(chunks, axis=axis)


def _squeeze_member(out: np.ndarray) -> np.ndarray:
    """Drop a leading singleton batch dimension from one member output."""
    return out[0] if out.ndim == 3 and out.shape[0] == 1 else out


def _detect_feature_mask(X) -> np.ndarray | None:
    """Boolean mask of all-NaN columns (used for SHAP-style masking)."""
    if hasattr(X, "columns"):
        mask = X.isna().all(axis=0).to_numpy()
    else:
        arr = np.asarray(X)
        if np.issubdtype(arr.dtype, np.number):
            mask = np.isnan(arr).all(axis=0)
        else:
            mask = np.array([np.isnan(arr[:, i]).all() for i in range(arr.shape[1])])
    if not np.any(mask):
        return None
    return mask


class _TabICLBase(BaseEstimator):
    """Shared checkpoint loading / backend plumbing."""

    # Assigned during fit(); declared for type checkers.
    model_config_: dict
    model_: Any
    X_encoder_: Any
    ensemble_generator_: Any
    model_kv_cache_: OrderedDict | None

    def __init__(
        self,
        *,
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        checkpoint_version: str = CLASSIFIER_CHECKPOINT,
        model_path=None,
        allow_auto_download: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
        backend: str = "auto",
        batch_size: int = 8,
        device=None,
        quantization: str = "none",
        quantization_granularity: str = "per_row",
    ) -> None:
        validate_quantization(quantization, quantization_granularity)
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend
        self.batch_size = batch_size
        self.device = device
        self.quantization = quantization
        self.quantization_granularity = quantization_granularity

    # -- backend handling ---------------------------------------------------

    def _resolve_backend(self) -> str:
        return get_tabicl_backend(self.backend)

    def _load_model(self, backend_name: str | None = None) -> None:
        if backend_name is None:
            backend_name = self._resolve_backend()
        self.backend_ = backend_name
        if self.device is not None and backend_name != "torch":
            raise ValueError(
                f"device={self.device!r} is only supported by the 'torch' "
                f"backend, but backend '{backend_name}' was selected."
            )
        _, config_dict, params = ensure_npz(
            filename=self.checkpoint_version,
            model_path=self.model_path,
            allow_auto_download=self.allow_auto_download,
        )
        if self.quantization == QUANTIZATION_TERNARY:
            params = _ternary_post_training_quantize(
                params, self.quantization_granularity
            )
            warnings.warn(
                "Ternary post-training quantization of TabICL checkpoints is "
                "experimental and can degrade accuracy substantially "
                "(attention Q/K/V projections are kept full precision).",
                UserWarning,
                stacklevel=2,
            )
        self.model_config_ = config_dict
        config = TabICLConfig.from_dict(config_dict)
        if backend_name == "torch":
            from ._tabicl._model_torch import TabICLTorchModel

            self.model_ = TabICLTorchModel(config, params, device=self.device)
        elif backend_name == "numpy":
            from ._tabicl._model_numpy import TabICLNumPyModel

            self.model_ = TabICLNumPyModel(config, params)
        elif backend_name == "mojo":
            from ._tabicl._mojo_backend import TabICLMojoModel

            self.model_ = TabICLMojoModel(config, params)
        else:  # pragma: no cover
            raise NotImplementedError(
                f"TabICL backend '{backend_name}' is not implemented."
            )

    @property
    def max_classes_(self) -> int:
        return int(self.model_.config.max_classes)

    # -- shared predict plumbing ---------------------------------------------

    def _chunks(self, n_test: int):
        """Yield ``(start, stop)`` slices of ``batch_size`` test rows."""
        step = self.batch_size if self.batch_size and self.batch_size > 0 else n_test
        for start in range(0, n_test, step):
            yield start, min(start + step, n_test)

    def _prepare_test_data(self, X, feature_mask):
        """Fill masked columns and run the numerical encoder."""
        if feature_mask is not None:
            if hasattr(X, "columns"):
                X.iloc[:, feature_mask] = 0.0
            else:
                X[:, feature_mask] = 0.0
        return self.X_encoder_.transform(X)


class TabICLClassifier(ClassifierMixin, _TabICLBase):
    """Tabular in-context learning classifier (TabICLv2).

    Fits nothing but preprocessing: predictions are produced in-context from
    the training set by a frozen TabICLv2 transformer, averaged over an
    ensemble of normalization / feature-order / class-order views.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members (normalization x shuffle views).
    norm_methods : str or list of str, optional
        Normalization methods among ``'none'``, ``'power'``, ``'quantile'``,
        ``'quantile_rtdl'``, ``'robust'``. Defaults to ``['none', 'power']``.
    feat_shuffle_method : {'latin', 'shift', 'random', 'none'}, default='latin'
        Feature permutation strategy for ensemble diversity.
    class_shuffle_method : {'shift', 'random', 'latin', 'none'}, default='shift'
        Class label permutation strategy.
    outlier_threshold : float, default=4.0
        Z-score threshold for soft outlier clipping.
    softmax_temperature : float, default=0.9
        Temperature applied when converting averaged logits to probabilities.
    average_logits : bool, default=True
        Average logits (then softmax) instead of probabilities.
    support_many_classes : bool, default=True
        Enable mixed-radix + hierarchical prediction when the number of
        classes exceeds the native maximum (10).
    kv_cache : bool, default=False
        Pre-compute key/value caches of the training data for faster repeated
        prediction. Not supported with more than the native number of classes.
    batch_size : int, default=8
        Number of test rows processed per forward pass. Does not affect
        predictions (test rows never attend to each other).
    device : str, optional
        Torch device for the ``'torch'`` backend (e.g. ``'cuda'``); ignored
        by the NumPy/Mojo backends.
    checkpoint_version : str, default='tabicl-classifier-v2-20260212.ckpt'
        Checkpoint file name in the ``jingang/TabICL`` HF repository.
    model_path : path-like, optional
        Local checkpoint path; skips downloading when it exists.
    allow_auto_download : bool, default=True
        Download the checkpoint when missing.
    random_state : int, optional
        Seed for ensemble shuffling and normalization noise.
    verbose : bool, default=False
    backend : {'auto', 'torch', 'numpy', 'mojo'}, default='auto'
        Compute backend; 'auto' honors ``SHINRIN_TABICL_BACKEND``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_classes_ : int
    """

    y_encoder_: Any
    classes_: np.ndarray
    n_classes_: int

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: int = 8,
        kv_cache: bool = False,
        checkpoint_version: str = CLASSIFIER_CHECKPOINT,
        model_path=None,
        allow_auto_download: bool = True,
        device=None,
        random_state: int | None = 42,
        verbose: bool = False,
        backend: str = "auto",
        quantization: str = "none",
        quantization_granularity: str = "per_row",
    ) -> None:
        super().__init__(
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            outlier_threshold=outlier_threshold,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            random_state=random_state,
            verbose=verbose,
            backend=backend,
            batch_size=batch_size,
            device=device,
            quantization=quantization,
            quantization_granularity=quantization_granularity,
        )
        self.n_estimators = n_estimators
        self.class_shuffle_method = class_shuffle_method
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.kv_cache = kv_cache

    def fit(self, X, y) -> TabICLClassifier:
        """Prepare encoders, ensemble views and (optionally) KV caches."""
        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        check_classification_targets(y)

        self._load_model()

        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > self.max_classes_ and self.kv_cache:
            raise ValueError(
                f"KV caching is not supported when the number of classes "
                f"({self.n_classes_}) exceeds the max number of classes "
                f"({self.max_classes_}) natively supported by the model."
            )
        if self.n_classes_ > self.max_classes_ and not self.support_many_classes:
            raise ValueError(
                f"The number of classes ({self.n_classes_}) exceeds the max "
                f"number ({self.max_classes_}) natively supported. Enable "
                "support_many_classes for mixed-radix/hierarchical prediction."
            )
        if (
            self.n_classes_ > self.max_classes_
            and self.support_many_classes
            and getattr(self, "backend_", None) == "mojo"
        ):
            # The Mojo kernels do not implement hierarchical many-class
            # prediction; transparently fall back to a backend that does.
            fallback = "torch" if importlib.util.find_spec("torch") else "numpy"
            warnings.warn(
                f"Mojo backend does not support {self.n_classes_} classes "
                f"(max {self.max_classes_}); falling back to the {fallback!r} "
                "backend for this fit.",
                stacklevel=2,
            )
            self._load_model(backend_name=fallback)

        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        self.ensemble_generator_ = EnsembleGenerator(
            classification=True,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y)

        self.model_kv_cache_: OrderedDict | None = None
        if self.kv_cache:
            self.model_kv_cache_ = OrderedDict()
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            for norm_method, (Xs, ys) in train_data.items():
                self.model_kv_cache_[norm_method] = [
                    self.model_.build_cache(Xs[i], ys[i]) for i in range(Xs.shape[0])
                ]
        return self

    def _member_outputs(self, X) -> np.ndarray:
        """Return stacked member outputs of shape (n_members, T_test, C)."""
        eg = self.ensemble_generator_
        has_cache = self.model_kv_cache_ is not None
        feature_mask = _detect_feature_mask(X)
        X = self._prepare_test_data(X, feature_mask)
        use_cache = has_cache and feature_mask is None

        outputs = []
        if use_cache:
            kv_caches = self.model_kv_cache_
            if kv_caches is None:  # pragma: no cover - defensive
                raise RuntimeError("kv_cache=True but no cache was built.")
            for norm_method, (Xs_test,) in eg.transform(X, mode="test").items():
                caches = kv_caches[norm_method]
                for i in range(Xs_test.shape[0]):
                    chunks = [
                        self.model_.predict_with_cache(
                            Xs_test[i, start:stop],
                            caches[i],
                            return_logits=self.average_logits,
                            temperature=self.softmax_temperature,
                        )
                        for start, stop in self._chunks(Xs_test.shape[1])
                    ]
                    outputs.append(_squeeze_member(_concat_chunks(chunks)))
        else:
            for Xs, ys in eg.transform(
                X, mode="both", feature_mask=feature_mask
            ).values():
                for i in range(Xs.shape[0]):
                    train_size = ys[i].shape[0]
                    R = self.model_.representations(Xs[i], ys[i])
                    chunks = []
                    for start, stop in self._chunks(Xs.shape[1] - train_size):
                        R_chunk = np.concatenate(
                            [
                                R[:, :train_size],
                                R[:, train_size + start : train_size + stop],
                            ],
                            axis=1,
                        )
                        chunks.append(
                            self.model_.predict_from_representations(
                                R_chunk,
                                ys[i],
                                return_logits=self.average_logits,
                                temperature=self.softmax_temperature,
                            )
                        )
                    outputs.append(_squeeze_member(_concat_chunks(chunks)))
        return np.stack(outputs, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        """Average class probabilities over the ensemble."""
        outputs = self._member_outputs(X)

        class_shuffles = []
        for shuffles in self.ensemble_generator_.class_shuffles_.values():
            class_shuffles.extend(shuffles)
        n_members = len(class_shuffles)
        if n_members != outputs.shape[0]:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Expected {n_members} ensemble outputs, got {outputs.shape[0]}."
            )

        avg = np.zeros_like(outputs[0])
        for out, shuffle in zip(outputs, class_shuffles):
            avg += out[..., shuffle]
        avg /= n_members

        if self.average_logits:
            avg = _softmax(avg, temperature=self.softmax_temperature)
        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Predict the most probable class label per sample."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TabICLRegressor(RegressorMixin, _TabICLBase):
    """Tabular in-context learning regressor (TabICLv2).

    Predictions are quantiles of the in-context predictive distribution;
    summary statistics (mean, median, arbitrary quantiles) are derived per
    ensemble member, mapped back to the original target scale and averaged.

    See :class:`TabICLClassifier` for shared parameters (minus
    ``class_shuffle_method`` / ``softmax_temperature`` / ``average_logits`` /
    ``support_many_classes``).
    """

    y_scaler_: Any

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: int = 8,
        kv_cache: bool = False,
        checkpoint_version: str = REGRESSOR_CHECKPOINT,
        model_path=None,
        allow_auto_download: bool = True,
        device=None,
        random_state: int | None = 42,
        verbose: bool = False,
        backend: str = "auto",
        quantization: str = "none",
        quantization_granularity: str = "per_row",
    ) -> None:
        super().__init__(
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            outlier_threshold=outlier_threshold,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            random_state=random_state,
            verbose=verbose,
            backend=backend,
            batch_size=batch_size,
            device=device,
            quantization=quantization,
            quantization_granularity=quantization_granularity,
        )
        self.n_estimators = n_estimators
        self.kv_cache = kv_cache

    def fit(self, X, y) -> TabICLRegressor:
        """Prepare encoders, ensemble views and (optionally) KV caches."""
        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)

        self._load_model()

        from sklearn.preprocessing import StandardScaler

        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        self.ensemble_generator_ = EnsembleGenerator(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y_scaled)

        self.model_kv_cache_: OrderedDict | None = None
        if self.kv_cache:
            self.model_kv_cache_ = OrderedDict()
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            for norm_method, (Xs, ys) in train_data.items():
                self.model_kv_cache_[norm_method] = [
                    self.model_.build_cache(Xs[i], ys[i]) for i in range(Xs.shape[0])
                ]
        return self

    def _member_quantiles(self, X) -> np.ndarray:
        """Stacked raw monotonic-ready decoder outputs (n_members, T_test, Q)."""
        eg = self.ensemble_generator_
        has_cache = self.model_kv_cache_ is not None
        feature_mask = _detect_feature_mask(X)
        X = self._prepare_test_data(X, feature_mask)
        use_cache = has_cache and feature_mask is None

        outputs = []
        if use_cache:
            kv_caches = self.model_kv_cache_
            if kv_caches is None:  # pragma: no cover - defensive
                raise RuntimeError("kv_cache=True but no cache was built.")
            for norm_method, (Xs_test,) in eg.transform(X, mode="test").items():
                caches = kv_caches[norm_method]
                for i in range(Xs_test.shape[0]):
                    chunks = [
                        self.model_.predict_with_cache(
                            Xs_test[i, start:stop], caches[i]
                        )
                        for start, stop in self._chunks(Xs_test.shape[1])
                    ]
                    outputs.append(_squeeze_member(_concat_chunks(chunks)))
        else:
            for Xs, ys in eg.transform(
                X, mode="both", feature_mask=feature_mask
            ).values():
                for i in range(Xs.shape[0]):
                    train_size = ys[i].shape[0]
                    R = self.model_.representations(Xs[i], ys[i])
                    chunks = []
                    for start, stop in self._chunks(Xs.shape[1] - train_size):
                        R_chunk = np.concatenate(
                            [
                                R[:, :train_size],
                                R[:, train_size + start : train_size + stop],
                            ],
                            axis=1,
                        )
                        chunks.append(
                            self.model_.predict_from_representations(R_chunk, ys[i])
                        )
                    outputs.append(_squeeze_member(_concat_chunks(chunks)))
        return np.stack(outputs, axis=0)

    def predict(
        self,
        X,
        output_type: str | list[str] = "mean",
        alphas: list[float] | None = None,
    ):
        """Predict target statistics on the original scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        output_type : str or list of str, default='mean'
            One of ``'mean'``, ``'median'``, ``'variance'``, ``'quantiles'``,
            ``'raw_quantiles'``; or a list of those.
        alphas : list of float, optional
            Probability levels for ``output_type='quantiles'``
            (default ``[0.1 ... 0.9]``).

        Returns
        -------
        ndarray or dict of ndarray
            ``(n_samples,)`` for scalar statistics, ``(n_samples, n_levels)``
            for quantile outputs; a dict when ``output_type`` is a list.
        """
        keys = [output_type] if isinstance(output_type, str) else list(output_type)
        quantile_out = self._member_quantiles(X)

        q2d = QuantileToDistribution(
            num_quantiles=int(self.model_.config.num_quantiles)
        )
        results: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for member in range(quantile_out.shape[0]):
            stats = q2d.stats(quantile_out[member], keys, alphas=alphas)
            for key in keys:
                results[key].append(stats[key])

        final = {}
        for key in keys:
            # Stack members so the final mean averages across the ensemble.
            stacked = np.stack(results[key], axis=0)
            shape = stacked.shape
            flat = self.y_scaler_.inverse_transform(
                np.asarray(stacked, dtype=float).reshape(-1, 1)
            )
            final[key] = flat.reshape(shape).mean(axis=0)

        return final[keys[0]] if len(keys) == 1 else final
