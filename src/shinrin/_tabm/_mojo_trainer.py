"""Adapter between the Python TabM trainer and the Mojo kernels.

The native module ``shinrin._native_tabm`` exposes a ``TabMTrainer`` bound
type constructed from a dims vector and per-feature bin counts. Its
methods operate directly on NumPy buffers:

- ``adam_epoch(parts)``: one shuffled minibatch Adam epoch (dropout included)
- ``lbfgs_minimize(parts)``: full-batch L-BFGS with backtracking line search
- ``loss_grad(parts)``: full-batch loss + gradient (parity testing)
- ``forward_avg(parts)``: k-member-averaged predictions into a preallocated array

``dims = [k, n_blocks, d_block, d_out, n_num_features, d_enc, d_cat,
use_embeddings, d_embedding]`` and ``bins`` holds the per-feature bin
counts.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ._backend import get_tabm_native
from ._layers import TabMConfig, TabMParams
from ._model import Batch

_LOCK = threading.Lock()
_TRAINERS: dict[tuple, Any] = {}


def _dims_vector(config: TabMConfig) -> np.ndarray:
    return np.array(
        [
            config.k,
            config.n_blocks,
            config.d_block,
            config.d_out,
            config.n_num_features,
            config.d_enc,
            sum(config.cat_cardinalities),
            1 if config.use_embeddings else 0,
            config.d_embedding,
        ],
        dtype=np.int64,
    )


def _bins_vector(config: TabMConfig) -> np.ndarray:
    return np.array(config.bin_counts, dtype=np.int64)


def _or_dummy(arr: np.ndarray | None) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def get_tabm_trainer(config: TabMConfig) -> Any:
    """Return a cached native ``TabMTrainer`` for the given configuration."""
    key = (
        config.k,
        config.n_blocks,
        config.d_block,
        config.d_out,
        config.n_num_features,
        tuple(config.cat_cardinalities),
        config.use_embeddings,
        config.d_embedding,
        tuple(config.bin_counts),
    )
    with _LOCK:
        trainer = _TRAINERS.get(key)
        if trainer is None:
            trainer = get_tabm_native().TabMTrainer(
                _dims_vector(config), _bins_vector(config)
            )
            _TRAINERS[key] = trainer
        return trainer


class NativeTrainer:
    """Thin wrapper providing a stable Python API over the Mojo kernels."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    @staticmethod
    def _data(batch: Batch, config: TabMConfig):
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

    def adam_epoch(
        self,
        theta: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        t: int,
        batch: Batch,
        params: TabMParams,
        space,
        *,
        lr: float,
        batch_size: int,
        dropout: float,
        alpha: float,
        seed: int,
        task: int = 0,
    ) -> tuple[float, int]:
        """Run one shuffled minibatch Adam epoch inside Mojo."""
        config = params.config
        assert config.arch_type == "tabm"
        x_num, x_enc, x_cat, y = self._data(batch, config)
        theta = np.ascontiguousarray(theta, dtype=np.float32)
        m = np.ascontiguousarray(m, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        loss, new_t = self._trainer.adam_epoch(
            [
                theta,
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
        space.scatter(theta, params)
        return float(loss), int(new_t)

    def forward_avg(
        self, theta: np.ndarray, batch: Batch, params: TabMParams
    ) -> np.ndarray:
        config = params.config
        x_num, x_enc, x_cat, _ = self._data(batch, config)
        out = np.zeros((len(x_num), config.d_out), dtype=np.float32)
        self._trainer.forward_avg(
            [np.ascontiguousarray(theta, dtype=np.float32), x_num, x_enc, x_cat, out]
        )
        return out

    def lbfgs(
        self,
        theta: np.ndarray,
        batch: Batch,
        params: TabMParams,
        space,
        *,
        max_iter: int,
        tol: float,
        alpha: float,
        history_size: int = 10,
        task: int = 0,
    ) -> tuple[int, list[float]]:
        config = params.config
        x_num, x_enc, x_cat, y = self._data(batch, config)
        theta = np.ascontiguousarray(theta, dtype=np.float32)
        losses = np.zeros(max_iter + 1, dtype=np.float64)
        n_iter = int(
            self._trainer.lbfgs_minimize(
                [
                    theta,
                    x_num,
                    x_enc,
                    x_cat,
                    y,
                    int(max_iter),
                    float(tol),
                    int(history_size),
                    float(alpha),
                    losses,
                    int(task),
                ]
            )
        )
        space.scatter(theta, params)
        return n_iter, [float(x) for x in losses[: n_iter + 1]]


def get_native_trainer(config: TabMConfig | None = None) -> NativeTrainer:
    """Return a :class:`NativeTrainer` bound to a cached native trainer."""
    if config is None:
        config = TabMConfig(
            n_num_features=1,
            cat_cardinalities=[],
            d_out=1,
            use_embeddings=False,
        )
    return NativeTrainer(get_tabm_trainer(config))
