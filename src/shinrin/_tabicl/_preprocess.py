"""Preprocessing pipeline and ensemble generation for TabICL.

NumPy ports of the upstream ``tabicl`` preprocessing stack
(``TransformToNumerical``, ``UniqueFeatureFilter``, ``OutlierRemover``,
``CustomStandardScaler``, ``RTDLQuantileTransformer``, ``PreprocessingPipeline``,
``Shuffler`` and ``EnsembleGenerator``) so that all backends share identical
data preparation semantics.
"""

from __future__ import annotations

import copy
import itertools
import random
from collections import OrderedDict
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "TransformToNumerical",
    "UniqueFeatureFilter",
    "OutlierRemover",
    "CustomStandardScaler",
    "RTDLQuantileTransformer",
    "PreprocessingPipeline",
    "Shuffler",
    "EnsembleGenerator",
]


def _is_dataframe(X: object) -> bool:
    return hasattr(X, "columns")


class TransformToNumerical(TransformerMixin, BaseEstimator):
    """Convert non-numeric DataFrame columns to numeric representations.

    DataFrames are handled with an ``OrdinalEncoder`` for categorical columns
    and a ``SimpleImputer`` for numeric ones; plain arrays must already be
    castable to float64 and are only imputed.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def fit(self, X, y=None):
        cat_tfm = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        num_tfm = SimpleImputer(keep_empty_features=True)

        if not _is_dataframe(X):
            X_arr = np.asarray(X)
            try:
                X_arr.astype(np.float64)
            except (ValueError, TypeError) as e:
                msg = (
                    "NumPy arrays passed to TabICL must be castable to a numeric "
                    f"dtype, but casting to float64 failed with: {e}. If your data "
                    "contains categorical or string columns, pass it as a pandas "
                    "DataFrame instead."
                )
                raise type(e)(msg) from None
            self.tfm_ = num_tfm
        else:
            cat_cols = make_column_selector(
                dtype_include=["string", "object", "category", "boolean"]
            )(X)
            cat_pos = [X.columns.get_loc(col) for col in cat_cols]
            numeric_cols = make_column_selector(dtype_include="number")(X)
            numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]
            self.tfm_ = ColumnTransformer(
                transformers=[
                    ("categorical", cat_tfm, cat_pos),
                    ("continuous", num_tfm, numeric_pos),
                ]
            )

        self.tfm_.fit(X)
        return self

    def transform(self, X):
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Drop features with at most ``threshold`` unique training values."""

    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        n_features = X.shape[1]
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(n_features, dtype=bool)
        else:
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(n_features)]
            )
        self.n_features_out_ = int(np.sum(self.features_to_keep_))
        return self

    def transform(self, X):
        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Two-stage z-score outlier clipping with log-based soft bounds."""

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    @staticmethod
    def _stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ddof = 1 if X.shape[0] > 1 else 0
        means = np.nanmean(X, axis=0)
        stds = np.maximum(np.nanstd(X, axis=0, ddof=ddof), 1e-6)
        return means, stds

    def fit(self, X, y=None):
        self.means_, self.stds_ = self._stats(X)
        lower = self.means_ - self.threshold * self.stds_
        upper = self.means_ + self.threshold * self.stds_
        outlier_mask = (X < lower) | (X > upper)
        X_clean = X.copy()
        X_clean[outlier_mask] = np.nan
        self.means_, self.stds_ = self._stats(X_clean)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Z-scaling by ``(x - mean) / (std + epsilon)`` clipped to ``[-100, 100]``."""

    def __init__(
        self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6
    ) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon
        return self

    def transform(self, X):
        is_vector = X.ndim == 1
        if is_vector:
            X = X.reshape(-1, 1)
        X_scaled = np.clip((X - self.mean_) / self.scale_, self.clip_min, self.clip_max)
        return X_scaled.reshape(-1) if is_vector else X_scaled

    def inverse_transform(self, X):
        is_vector = X.ndim == 1
        if is_vector:
            X = X.reshape(-1, 1)
        X_out = X * self.scale_ + self.mean_
        return X_out.reshape(-1) if is_vector else X_out


class RTDLQuantileTransformer(TransformerMixin, BaseEstimator):
    """Quantile transform with RTDL-style noise and adaptive quantile count."""

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ) -> None:
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        if self.noise > 0:
            stds = np.std(X, axis=0, keepdims=True)
            noise_std = self.noise / np.maximum(stds, self.noise)
            rng = np.random.default_rng(self.random_state)
            X = X + noise_std * rng.standard_normal(X.shape)
        normalizer.fit(X)
        self.normalizer_ = normalizer
        return self

    def transform(self, X, y=None):
        return self.normalizer_.transform(X)


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """CustomStandardScaler -> optional normalizer -> outlier clipping."""

    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )
            elif self.normalization_method == "quantile_rtdl":
                from sklearn.pipeline import Pipeline

                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(
                                output_distribution="normal",
                                random_state=self.random_state,
                            ),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalization_method}"
                )

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)
        return self

    def transform(self, X):
        X = self.standard_scaler_.transform(X)
        if self.normalizer_ is not None:
            try:
                X = self.normalizer_.transform(X)
            except ValueError:
                # Rare unseen outliers can break quantile transforms; clip to
                # the training range seen after standard scaling and retry.
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        X = self.outlier_remover_.transform(X)
        return X


class Shuffler:
    """Generate index permutations ('none'/'shift'/'random'/'latin')."""

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> list[list[int]]:
        rng = random.Random(self.random_state)
        indices = list(range(self.n_elements))

        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

        if method == "none" or n_estimators == 1:
            return [indices]

        if method == "shift":
            return [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        if method == "random":
            if self.n_elements <= 5:
                all_perms = [list(p) for p in itertools.permutations(indices)]
                return rng.sample(all_perms, min(n_estimators, len(all_perms)))
            return [rng.sample(indices, self.n_elements) for _ in range(n_estimators)]
        if method == "latin":
            square = self._latin_squares(rng)
            return square
        raise ValueError(
            f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'."
        )

    def _shuffle_transpose_shuffle(self, matrix, rng: random.Random):
        square = copy.deepcopy(matrix)
        rng.shuffle(square)
        trans = list(zip(*square))
        rng.shuffle(trans)
        return trans

    def _rls(self, symbols: list[int], rng: random.Random) -> list[list[int]]:
        """Recursive latin-square construction (Bose-style)."""
        if len(symbols) == 1:
            return [symbols]
        sym = rng.choice(symbols)
        symbols.remove(sym)
        square = self._rls(symbols, rng)
        square.append(square[0].copy())
        for i in range(len(square)):
            square[i].insert(i, sym)
        return square

    def _latin_squares(self, rng: random.Random) -> list[list[int]]:
        square = self._rls(list(range(self.n_elements)), rng)
        shuffles = self._shuffle_transpose_shuffle(square, rng)
        return [list(row) for row in shuffles]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Create normalization x feature-shuffle x class-shuffle ensemble views.

    ``transform(mode='both')`` returns an ``OrderedDict`` mapping each
    normalization method to ``(Xs, ys)`` stacks of shape
    ``(n_members, n_train + n_test, n_features)`` and ``(n_members, n_train)``.
    """

    def __init__(
        self,
        classification: bool,
        n_estimators: int,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        self.norm_methods_ = (
            ["none", "power"]
            if self.norm_methods is None
            else ([self.norm_methods] if isinstance(self.norm_methods, str) else list(self.norm_methods))
        )

        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        if self.classification:
            self.n_classes_ = len(np.unique(y))

        self.ensemble_configs_, self.feature_shuffles_, y_patterns = (
            self._generate_ensemble()
        )
        if self.classification:
            self.class_shuffles_ = y_patterns

        self.preprocessors_: dict[str, PreprocessingPipeline] = {}
        for norm_method in self.ensemble_configs_:
            preprocessor = PreprocessingPipeline(
                normalization_method=norm_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            preprocessor.fit(X)
            self.preprocessors_[norm_method] = preprocessor
        return self

    def _generate_ensemble(self):
        feat_shuffler = Shuffler(
            n_elements=self.n_features_in_,
            method=self.feat_shuffle_method,
            random_state=self.random_state,
        )
        X_shuffles = feat_shuffler.shuffle(self.n_estimators)

        if self.classification:
            class_shuffler = Shuffler(
                n_elements=self.n_classes_,
                method=self.class_shuffle_method,
                random_state=self.random_state,
            )
            y_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            y_patterns = [None]

        shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
        random.Random(self.random_state).shuffle(shuffle_configs)

        shuffle_norm_configs = list(itertools.product(shuffle_configs, self.norm_methods_))
        shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators]

        used_methods = list({config[1] for config in shuffle_norm_configs})

        ensemble_configs: OrderedDict[str, list] = OrderedDict()
        X_shuffle_dict: OrderedDict[str, list] = OrderedDict()
        y_pattern_dict: OrderedDict[str, list] = OrderedDict()
        for method in used_methods:
            configs = [config[0] for config in shuffle_norm_configs if config[1] == method]
            X_shuffle_dict[method] = [config[0] for config in configs]
            y_pattern_dict[method] = [config[1] for config in configs]
            ensemble_configs[method] = configs
        return ensemble_configs, X_shuffle_dict, y_pattern_dict

    def _masked_shuffle_maps(self, feature_mask: np.ndarray):
        filtered_mask = feature_mask[self.unique_filter_.features_to_keep_]
        kept_cols = ~filtered_mask
        idx_map: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(len(filtered_mask)):
            if kept_cols[old_idx]:
                idx_map[old_idx] = new_idx
                new_idx += 1
        masked_shuffles: OrderedDict[str, list] = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            remapped = []
            for feat_shuffle, _ in shuffle_configs:
                remapped.append([idx_map[i] for i in feat_shuffle if i in idx_map])
            masked_shuffles[norm_method] = remapped
        return filtered_mask, kept_cols, masked_shuffles

    def transform(self, X=None, mode: str = "both", feature_mask=None):
        if mode not in ("both", "train", "test"):
            raise ValueError(f"Invalid mode: {mode}")

        if feature_mask is not None:
            filtered_mask, kept_cols, masked_shuffles = self._masked_shuffle_maps(
                feature_mask
            )

        if mode == "train":
            y = self.y_
            data: OrderedDict = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_pp = self.preprocessors_[norm_method].X_transformed_
                if feature_mask is not None:
                    X_pp = X_pp[:, kept_cols]
                X_ensemble, y_ensemble = [], []
                for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                    if feature_mask is not None:
                        feat_shuffle = masked_shuffles[norm_method][i]
                    X_ensemble.append(X_pp[:, feat_shuffle])
                    if self.classification:
                        y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                    else:
                        y_ensemble.append(y)
                data[norm_method] = (
                    np.stack(X_ensemble, axis=0),
                    np.stack(y_ensemble, axis=0),
                )
            return data

        assert X is not None, "X is required when mode is 'test' or 'both'"
        X = self.unique_filter_.transform(X)
        if feature_mask is not None:
            X = np.array(X, dtype=np.float64)
            X[:, filtered_mask] = 0.0

        if mode == "test":
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_test_pp = self.preprocessors_[norm_method].transform(X)
                if feature_mask is not None:
                    X_test_pp = X_test_pp[:, kept_cols]
                X_ensemble = []
                for i, (feat_shuffle, _) in enumerate(shuffle_configs):
                    if feature_mask is not None:
                        feat_shuffle = masked_shuffles[norm_method][i]
                    X_ensemble.append(X_test_pp[:, feat_shuffle])
                data[norm_method] = (np.stack(X_ensemble, axis=0),)
            return data

        # mode == "both"
        y = self.y_
        data = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            preprocessor = self.preprocessors_[norm_method]
            X_train_pp = preprocessor.X_transformed_
            X_test_pp = preprocessor.transform(X)
            if feature_mask is not None:
                X_train_pp = X_train_pp[:, kept_cols]
                X_test_pp = X_test_pp[:, kept_cols]
            X_variant = np.concatenate([X_train_pp, X_test_pp], axis=0)
            X_ensemble, y_ensemble = [], []
            for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                if feature_mask is not None:
                    feat_shuffle = masked_shuffles[norm_method][i]
                X_ensemble.append(X_variant[:, feat_shuffle])
                if self.classification:
                    y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                else:
                    y_ensemble.append(y)
            data[norm_method] = (
                np.stack(X_ensemble, axis=0),
                np.stack(y_ensemble, axis=0),
            )
        return data
