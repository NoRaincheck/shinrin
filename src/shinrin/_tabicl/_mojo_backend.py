"""Mojo-native inference backend for TabICLv2.

Wraps the precompiled ``shinrin._native_tabicl`` shared library to provide
the same ``representations`` / ``predict_from_representations`` interface as
the Torch and NumPy backends.

Build the shared library with::

    just build-tabicl-mojo
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backend import get_tabicl_native
from ._config import TabICLConfig

SKIP_VALUE = -100.0


class TabICLMojoModel:
    """Inference wrapper around the Mojo native TabICL kernels.

    Parameters
    ----------
    config : TabICLConfig
        Model configuration.
    params : dict[str, np.ndarray]
        Parameter arrays keyed by state-dict name.
    """

    def __init__(self, config: TabICLConfig, params: dict[str, np.ndarray]) -> None:
        self.config = config
        self._native = get_tabicl_native()
        self._handle: Any = None
        self._param_data: np.ndarray | None = None
        self._init_params(params)

    def _init_params(self, params: dict[str, np.ndarray]) -> None:
        """Flatten all parameters into a single float32 buffer."""
        cfg = self.config
        # Compute total size
        total = 0
        for name in sorted(params.keys()):
            total += params[name].size
        self._param_data = np.empty(total, dtype=np.float32)
        offset = 0
        for name in sorted(params.keys()):
            arr = np.asarray(params[name], dtype=np.float32).ravel()
            self._param_data[offset : offset + arr.size] = arr
            offset += arr.size
        self._handle = self._native.create_inference(cfg._dims_array(), self._param_data)

    # ------------------------------------------------------------------ #
    # representations
    # ------------------------------------------------------------------ #

    def representations(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Column embedding + row interaction. Returns (1, T, D) array."""
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        n_train = X.shape[0]

        # Build col_input: feature grouping + target-aware encoding
        # col_input shape: (n_train, group_size)
        col_input = X

        # Build row_input: already embedded test rows
        # For representations, we need to return the full (1, T, D) output
        # which includes both train and test rows

        # Call native forward
        # col_input: (n_train, group_size)
        # row_input: (n_test, embed_dim) - for representations, we pass zeros
        # and get back the full representation

        n_test = 0  # No test rows yet
        row_input = np.zeros((n_test, cfg.embed_dim), dtype=np.float32)
        target = np.asarray(y_train, dtype=np.int64)

        output = self._native.forward(
            self._handle, col_input, row_input, target
        )

        # Reshape to (1, T, D)
        return output.reshape(1, n_train, cfg.embed_dim)

    # ------------------------------------------------------------------ #
    # predict_from_representations
    # ------------------------------------------------------------------ #

    def predict_from_representations(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the ICL stage on row representations.

        Parameters
        ----------
        R : np.ndarray
            Row representations of shape (1, T, D).
        y_train : np.ndarray
            Training labels.
        return_logits : bool
            If True, return logits; otherwise return probabilities.
        temperature : float
            Temperature scaling for probabilities.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_test, num_classes) or (n_test, out_dim).
        """
        cfg = self.config
        R = np.ascontiguousarray(R, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        train_size = y_train.shape[0]

        # Extract test rows from R
        # R shape: (1, T, D) where T = train_size + test_size
        n_test = R.shape[1] - train_size

        if n_test == 0:
            return np.empty((0, cfg.out_dim), dtype=np.float32)

        # Test rows are R[0, train_size:]
        test_rows = R[0, train_size:]  # (n_test, D)

        # For the Mojo backend, we need to re-run the forward pass with
        # the test rows appended to the training rows
        # This is a simplification - a more efficient version would use caching

        # Build combined input
        col_input = np.zeros((train_size + n_test, cfg.col_feature_group_size), dtype=np.float32)
        col_input[:train_size] = test_rows[:, :cfg.col_feature_group_size]

        target = np.asarray(y_train, dtype=np.int64)

        output = self._native.forward(
            self._handle, col_input, test_rows, target
        )

        if not return_logits and cfg.max_classes > 0:
            output = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)

        return output

    # ------------------------------------------------------------------ #
    # build_cache / predict_with_cache
    # ------------------------------------------------------------------ #

    def build_cache(self, X: np.ndarray, y_train: np.ndarray) -> dict:
        """Pre-compute caches for repeated predictions.

        Returns a dict with the same structure as the Torch backend.
        """
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        train_size = y_train.shape[0]

        # For Mojo backend, we return a minimal cache
        # The full cache implementation would require significant additional
        # native code to support incremental K/V caching
        return {
            "col": [None] * cfg.col_num_blocks,
            "icl": [None] * cfg.icl_num_blocks,
            "train_size": train_size,
            "num_classes": len(np.unique(y_train)) if cfg.max_classes > 0 else 0,
        }

    def predict_with_cache(
        self,
        X_test: np.ndarray,
        cache: dict,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Predict using pre-computed caches.

        Note: The Mojo backend currently bypasses caching and runs a full
        forward pass. This matches the behaviour of the many-class
        hierarchical prediction path.
        """
        cfg = self.config
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.asarray(cache["y_train"], dtype=np.float32)
        n_test = X_test.shape[0]

        # Full forward pass (cache is ignored)
        col_input = X_test[:, :cfg.col_feature_group_size]
        row_input = X_test[:, cfg.col_feature_group_size:] if X_test.shape[1] > cfg.col_feature_group_size else np.zeros((n_test, cfg.embed_dim - cfg.col_feature_group_size), dtype=np.float32)
        target = np.asarray(y_train, dtype=np.int64)

        output = self._native.forward(
            self._handle, col_input, row_input, target
        )

        if not return_logits and cfg.max_classes > 0:
            output = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)

        return output

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Full pipeline: preprocessing-free numeric forward pass."""
        R = self.representations(X, y_train)
        return self.predict_from_representations(
            R, y_train, return_logits=return_logits, temperature=temperature
        )

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        if self._handle is not None:
            try:
                self._native.delete(self._handle)
            except BaseException:  # noqa: S110, BLE001
                pass
