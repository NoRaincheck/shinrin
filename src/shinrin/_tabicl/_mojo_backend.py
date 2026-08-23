"""Mojo-native inference backend for TabICLv2 (**experimental**).

Wraps the precompiled ``shinrin._native_tabicl`` shared library following the
same pattern as the TabM trainer bindings: the Python side packs the
architecture hyper-parameters (:meth:`TabICLConfig.dims_array`) and the state
dict into contiguous buffers, then drives a bound ``TabICLInference`` type.

.. note::
   Inference runs as three staged kernel calls — ``stage_col`` (column
   embedding), ``stage_row`` (row interaction) and
   ``predict_from_representations`` (in-context learning + decoder) —
   composed by :meth:`TabICLMojoModel.forward`. KV-cache methods are not
   implemented yet.

Build the shared library with::

    just build-tabicl-mojo
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backend import get_tabicl_native
from ._config import TabICLConfig
from ._mojo_layout import canonical_tensor_specs, pack_params

SKIP_VALUE = -100.0


def expected_layout_offsets(config: TabICLConfig) -> list[int]:
    """Offsets the native kernel's layout walk must produce for ``config``.

    Derived purely from :func:`canonical_tensor_specs` and compared against
    ``TabICLInference.layout_offsets()`` at construction time. Element order
    matches ``_tabicl_kernels.mojo``; bias-free LayerNorms and disabled
    y-encoders report ``0`` on both sides.
    """
    off: dict[str, int] = {}
    cur = 0
    for name, shape in canonical_tensor_specs(config):
        off[name] = cur
        cur += int(np.prod(shape, dtype=np.int64))

    out = [
        cur,
        off["col_embedder.in_linear.weight"],
        off["col_embedder.in_linear.bias"],
    ]
    if config.col_target_aware:
        out += [
            off["col_embedder.y_encoder.weight"],
            off["col_embedder.y_encoder.bias"],
        ]
    else:
        out += [0, 0]

    b0 = off["col_embedder.tf_col.blocks.0.ind_vectors"]
    a1 = off["col_embedder.tf_col.blocks.0.multihead_attn1.attn.in_proj_weight"]
    a2 = off["col_embedder.tf_col.blocks.0.multihead_attn2.attn.in_proj_weight"]
    cls = off["row_interactor.cls_tokens"]
    # attn1 spans [a1, a2); one block spans [ind_vectors, next ind_vectors).
    if config.col_num_blocks > 1:
        b1 = off["col_embedder.tf_col.blocks.1.ind_vectors"]
        stride = b1 - b0
    else:
        stride = cls - b0
    out += [
        b0,
        stride,
        a2 - a1,  # attn1 section size
    ]

    r0 = off["row_interactor.tf_row.blocks.0.attn.in_proj_weight"]
    ln0 = off["icl_predictor.ln.weight"]
    out += [
        cls,
        off["row_interactor.out_ln.weight"],
        off.get("row_interactor.out_ln.bias", 0),
        off["row_interactor.tf_row.rope.freqs"],
        r0,
        (ln0 - r0) // config.row_num_blocks,  # row block size
        ln0,
        off.get("icl_predictor.ln.bias", 0),
        off["icl_predictor.y_encoder.weight"],
        off["icl_predictor.y_encoder.bias"],
        off["icl_predictor.decoder.0.weight"],
        off["icl_predictor.decoder.0.bias"],
        off["icl_predictor.decoder.2.weight"],
        off["icl_predictor.decoder.2.bias"],
    ]
    i0 = off["icl_predictor.tf_icl.blocks.0.attn.in_proj_weight"]
    # The icl blocks are the final section: they span [i0, total).
    return out + [
        i0,
        (out[0] - i0) // config.icl_num_blocks,  # icl block size
    ]


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
        """Pack ``params`` in canonical layout order and build the native model.

        Raises
        ------
        KeyError
            If the state dict is missing a tensor required by the layout.
        ValueError
            If the state dict has unexpected tensors or shape mismatches.
        RuntimeError
            If the native kernel's internal offset walk disagrees with the
            packed buffer length (layout drift between the two sides).
        """
        self.config = config
        # Canonical-order packing (single source of truth:
        # ``_mojo_layout.canonical_tensor_specs``). Missing/unknown/mis-shaped
        # tensors fail here instead of silently misaligning weights.
        self._param_data = pack_params(config, params)
        handle = get_tabicl_native().TabICLInference(
            config.dims_array(), self._param_data
        )
        expected = int(self._param_data.size)
        got = int(handle.param_count())
        if got != expected:
            raise RuntimeError(
                "native TabICL parameter walk disagrees with the canonical "
                f"layout: kernel walked {got} floats but the packed buffer "
                f"holds {expected}; update ``_mojo_layout.py`` and "
                "``_tabicl_kernels.mojo`` together"
            )
        # Exact structural fingerprint: every offset the kernel walked must
        # equal the offset derived from the canonical spec, so ordering or
        # sizing drift between the two sides fails loudly right here.
        raw_offsets = handle.layout_offsets()
        # ``Python.list`` wraps the payload once; normalize either shape.
        if len(raw_offsets) == 1 and hasattr(raw_offsets[0], "__len__"):
            raw_offsets = raw_offsets[0]
        got_offsets = [int(x) for x in raw_offsets]
        want_offsets = expected_layout_offsets(config)
        if got_offsets != want_offsets:
            raise RuntimeError(
                "native TabICL layout walk diverged from the canonical spec: "
                f"kernel offsets {got_offsets} vs spec offsets {want_offsets}; "
                "update ``_mojo_layout.py`` and ``_tabicl_kernels.mojo`` "
                "together"
            )
        self._handle: Any = handle

    # ------------------------------------------------------------------ #
    # end-to-end forward
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_target(cfg: TabICLConfig, y_train: np.ndarray) -> np.ndarray:
        """Canonical target buffer for the kernels: int64 labels for
        classification, float32 values for regression."""
        if cfg.max_classes > 0:
            return np.ascontiguousarray(y_train, dtype=np.int64)
        return np.ascontiguousarray(y_train, dtype=np.float32)

    def forward(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the native end-to-end forward pass (staged composition).

        Parameters
        ----------
        X : np.ndarray of shape (n_train + n_test, n_features)
            Numeric training rows followed by test rows.
        y_train : np.ndarray of shape (n_train,)
            Integer class labels (or scaled regression targets).
        return_logits : bool
            Return raw logits instead of probabilities.
        temperature : float
            Softmax temperature applied when ``return_logits=False``.
        """
        cfg = self.config
        x = np.ascontiguousarray(X, dtype=np.float32)
        y = self._prepare_target(cfg, np.asarray(y_train))
        reps = self.representations(x, y)
        return self.predict_from_representations(
            reps, y, return_logits=return_logits, temperature=temperature
        )

    # ------------------------------------------------------------------ #
    # staged API
    # ------------------------------------------------------------------ #

    def representations(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Column embedding + row interaction. Returns (1, T, D) array."""
        cfg = self.config
        x = np.ascontiguousarray(X, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-D (rows, features); got {x.shape}")
        target = self._prepare_target(cfg, np.asarray(y_train))
        if target.ndim != 1:
            raise ValueError("y_train must be 1-D")
        n_rows, n_features = x.shape
        train_size = target.shape[0]
        if not 0 < train_size <= n_rows:
            raise ValueError(f"train_size {train_size} must be within [1, {n_rows}]")

        g_total = cfg.row_num_cls + n_features
        col = np.asarray(self._handle.stage_col([x, target]), dtype=np.float32)
        expected = n_rows * g_total * cfg.embed_dim
        if col.size != expected:
            raise ValueError(
                f"stage_col returned {col.size} floats, expected {expected}"
            )

        reps = np.asarray(
            self._handle.stage_row(
                [
                    col.reshape(n_rows, g_total * cfg.embed_dim),
                    np.array([g_total], dtype=np.int64),
                ]
            ),
            dtype=np.float32,
        )
        expected = n_rows * cfg.icl_dim
        if reps.size != expected:
            raise ValueError(
                f"stage_row returned {reps.size} floats, expected {expected}"
            )
        return reps.reshape(1, n_rows, cfg.icl_dim)

    def predict_from_representations(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the ICL stage on row representations.

        Returns ``(test_size, num_classes)`` logits/probabilities for
        classification or ``(test_size, out_dim)`` raw quantiles for
        regression.
        """
        cfg = self.config
        # The kernel y-encodes the train prefix in place; give it a private
        # copy so caller-visible ``R`` is never mutated.
        r = np.array(R, dtype=np.float32, copy=True, order="C").reshape(-1, cfg.icl_dim)
        y = self._prepare_target(cfg, np.asarray(y_train))
        if y.ndim != 1:
            raise ValueError("y_train must be 1-D")
        train_size = y.shape[0]
        if not 0 < train_size <= r.shape[0]:
            raise ValueError(
                f"train_size {train_size} must be within [1, {r.shape[0]}]"
            )

        if cfg.max_classes > 0 and len(np.unique(y_train)) > cfg.max_classes:
            raise NotImplementedError(
                "many-class hierarchical prediction is not supported by the "
                "Mojo backend yet; use the 'numpy'/'torch' backends"
            )

        out_all = np.asarray(
            self._handle.predict_from_representations([r, y]), dtype=np.float32
        ).reshape(r.shape[0], cfg.out_dim)
        out = out_all[train_size:]

        if cfg.max_classes == 0:
            return out
        logits = out[:, : len(np.unique(y_train))]
        if return_logits:
            return logits
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs

    def build_cache(self, X: np.ndarray, y_train: np.ndarray) -> dict:
        """Pre-compute col-stage and ICL-stage K/V caches natively.

        Returns a dict with ``"col"`` shaped ``(G, blocks, 2, n_inds,
        embed_dim)`` and ``"icl"`` shaped ``(blocks, 2, train_size,
        icl_dim)`` float32 views over native flat buffers (``kv`` axis:
        0=key, 1=value), plus ``"train_size"``/``"num_classes"`` metadata.
        The caches are backend-specific; pass them back only to this
        backend's :meth:`predict_with_cache`.
        """
        cfg = self.config
        x = np.ascontiguousarray(X, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-D (rows, features); got {x.shape}")
        y = self._prepare_target(cfg, np.asarray(y_train))
        if y.ndim != 1:
            raise ValueError("y_train must be 1-D")
        train_size = y.shape[0]
        if train_size < 1:
            raise ValueError("build_cache needs at least one training row")

        col_flat, icl_flat = self._handle.build_cache([x, y])
        col_flat = np.asarray(col_flat, dtype=np.float32)
        icl_flat = np.asarray(icl_flat, dtype=np.float32)

        g_total = cfg.row_num_cls + x.shape[1]
        col_shape = (
            g_total,
            cfg.col_num_blocks,
            2,
            cfg.col_num_inds,
            cfg.embed_dim,
        )
        icl_shape = (cfg.icl_num_blocks, 2, train_size, cfg.icl_dim)
        if col_flat.size != int(np.prod(col_shape)):
            raise ValueError(
                f"build_cache returned {col_flat.size} col floats, "
                f"expected {int(np.prod(col_shape))}"
            )
        if icl_flat.size != int(np.prod(icl_shape)):
            raise ValueError(
                f"build_cache returned {icl_flat.size} icl floats, "
                f"expected {int(np.prod(icl_shape))}"
            )
        num_classes = len(np.unique(y_train)) if cfg.max_classes > 0 else 0
        return {
            "col": col_flat.reshape(col_shape),
            "icl": icl_flat.reshape(icl_shape),
            "train_size": train_size,
            "num_classes": num_classes,
        }

    def predict_with_cache(
        self,
        X_test: np.ndarray,
        cache: dict,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Predict test rows against a native :meth:`build_cache` result."""
        cfg = self.config
        x = np.ascontiguousarray(X_test, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"X_test must be 2-D (rows, features); got {x.shape}")
        train_size = int(cache["train_size"])
        if train_size < 1:
            raise ValueError("cache train_size must be at least 1")

        # The kernels only read the target's length here (targets were
        # already folded into the cache at build time).
        sentinel_dtype = np.int64 if cfg.max_classes > 0 else np.float32
        target = np.zeros(train_size, dtype=sentinel_dtype)
        col = np.ascontiguousarray(cache["col"], dtype=np.float32).reshape(-1)
        icl = np.ascontiguousarray(cache["icl"], dtype=np.float32).reshape(-1)

        out_all = np.asarray(
            self._handle.predict_with_cache([x, target, col, icl]),
            dtype=np.float32,
        )
        out = out_all.reshape(x.shape[0], cfg.out_dim)
        if cfg.max_classes == 0:
            return out
        num_classes = int(cache.get("num_classes", cfg.max_classes))
        logits = out[:, :num_classes]
        if return_logits:
            return logits
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #

    @property
    def param_count(self) -> int:
        """Number of float32 parameters held by the native instance."""
        return int(self._handle.param_count())

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        handle = getattr(self, "_handle", None)
        free = getattr(handle, "unsafe_free", None)
        if free is None:
            return
        try:
            free()
        except BaseException:  # noqa: BLE001, S110 - best-effort cleanup
            pass
