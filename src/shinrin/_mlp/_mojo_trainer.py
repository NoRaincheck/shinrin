"""Adapter between the Python MLP trainer and the Mojo kernels.

The native module ``shinrin._native_mlp`` exposes an ``MLPTrainer`` bound
type constructed from a dims vector, the layer-size table and per-feature
bin counts. Its methods operate directly on NumPy buffers:

- ``adam_epoch(parts)``: one shuffled minibatch Adam epoch (dropout included)
- ``lbfgs_minimize(parts)``: full-batch L-BFGS with backtracking line search
- ``loss_grad(parts)``: full-batch loss + gradient (parity testing)
- ``forward(parts)``: predictions into a preallocated array

``dims = [n_num_features, d_enc, d_cat, use_embeddings, d_embedding,
activation_code]``, ``layersizes`` holds ``[d_in, h1, ..., d_out]`` and
``bins`` holds the per-feature bin counts.

All methods mutate ``theta`` in place; callers keep parameter arrays bound
to views of ``theta`` (see ``FlatSpace.scatter``) so no re-scatter is
needed after native updates.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ._backend import get_mlp_native
from ._layers import MLPConfig

_LOCK = threading.Lock()
_TRAINERS: dict[tuple, Any] = {}

_ACT_CODES = {"identity": 0, "logistic": 1, "tanh": 2, "relu": 3}


def _dims_vector(config: MLPConfig) -> np.ndarray:
    return np.array(
        [
            config.n_num_features,
            config.d_enc,
            sum(config.cat_cardinalities),
            1 if config.use_embeddings else 0,
            config.d_embedding,
            _ACT_CODES[config.activation],
        ],
        dtype=np.int64,
    )


def _or_dummy(arr: np.ndarray | None) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def get_native_trainer(config: MLPConfig) -> NativeTrainer:
    """Return a cached native ``MLPTrainer`` wrapper for the configuration."""
    key = (
        config.n_num_features,
        config.d_enc,
        tuple(config.cat_cardinalities),
        config.use_embeddings,
        config.d_embedding,
        config.activation,
        tuple(config.layer_sizes),
        tuple(config.bin_counts),
    )
    with _LOCK:
        trainer = _TRAINERS.get(key)
        if trainer is None:
            layersizes = np.array(config.layer_sizes, dtype=np.int64)
            bins = np.array(config.bin_counts, dtype=np.int64)
            trainer = NativeTrainer(
                get_mlp_native().MLPTrainer(_dims_vector(config), layersizes, bins)
            )
            _TRAINERS[key] = trainer
        return trainer


class NativeTrainer:
    """Thin wrapper providing a stable Python API over the Mojo kernels."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    @property
    def param_count(self) -> int:
        return int(self._trainer.param_count())

    @staticmethod
    def _data(batch: BatchLike, config: MLPConfig):
        x_num = _or_dummy(batch.x_num)
        x_enc = (
            _or_dummy(batch.x_enc)
            if config.use_embeddings and config.n_num_features
            else np.zeros(1, dtype=np.float32)
        )
        x_cat = _or_dummy(batch.x_cat)
        y = np.asarray(batch.y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        return x_num, x_enc, x_cat, np.ascontiguousarray(y)

    def loss_grad(
        self,
        theta: np.ndarray,
        batch: BatchLike,
        config: MLPConfig,
        task: int = 0,
        alpha: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """Full-batch loss + gradient (used by parity tests)."""
        x_num, x_enc, x_cat, y = self._data(batch, config)
        loss, grad = self._trainer.loss_grad(
            [np.ascontiguousarray(theta, dtype=np.float32), x_num, x_enc, x_cat, y,
             int(task), float(alpha)]
        )
        return float(loss), np.asarray(grad)

    def forward(
        self,
        theta: np.ndarray,
        batch: BatchLike,
        config: MLPConfig,
        out: np.ndarray,
    ) -> None:
        """Write predictions ``(N, d_out)`` into the preallocated ``out``."""
        x_num, x_enc, x_cat, _ = self._data(batch, config)
        self._trainer.forward(
            [np.ascontiguousarray(theta, dtype=np.float32), x_num, x_enc, x_cat, out]
        )

    def adam_epoch(
        self,
        theta: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        t: int,
        batch: BatchLike,
        config: MLPConfig,
        *,
        lr: float,
        batch_size: int,
        dropout: float,
        alpha: float,
        seed: int,
        task: int,
    ) -> tuple[float, int]:
        """One shuffled minibatch Adam epoch; returns ``(loss, t_new)``."""
        x_num, x_enc, x_cat, y = self._data(batch, config)
        loss, t_new = self._trainer.adam_epoch(
            [
                np.ascontiguousarray(theta, dtype=np.float32),
                m,
                v,
                int(t),
                x_num,
                x_enc,
                x_cat,
                y,
                float(lr),
                int(batch_size),
                float(dropout),
                float(alpha),
                int(seed),
                int(task),
            ]
        )
        return float(loss), int(t_new)

    def lbfgs(
        self,
        theta: np.ndarray,
        batch: BatchLike,
        config: MLPConfig,
        *,
        max_iter: int,
        tol: float,
        alpha: float,
        task: int,
    ) -> tuple[int, list[float]]:
        """Full-batch L-BFGS; returns ``(nit, losses)``."""
        x_num, x_enc, x_cat, y = self._data(batch, config)
        losses = np.zeros(max_iter + 2, dtype=np.float64)
        nit = self._trainer.lbfgs_minimize(
            [
                np.ascontiguousarray(theta, dtype=np.float32),
                x_num,
                x_enc,
                x_cat,
                y,
                int(max_iter),
                float(tol),
                10,
                float(alpha),
                losses,
                int(task),
            ]
        )
        return int(nit), [float(x) for x in losses[: nit + 1]]


class BatchLike:
    """Structural type: anything exposing x_num/x_enc/x_cat/y arrays."""
