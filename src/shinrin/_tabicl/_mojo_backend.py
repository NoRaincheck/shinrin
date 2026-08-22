"""Mojo-native inference backend for TabICLv2 (**experimental**).

Wraps the precompiled ``shinrin._native_tabicl`` shared library following the
same pattern as the TabM trainer bindings: the Python side packs the
architecture hyper-parameters (:meth:`TabICLConfig.dims_array`) and the state
dict into contiguous buffers, then drives a bound ``TabICLInference`` type.

.. note::
   The kernels currently expose a single end-to-end :meth:`TabICLMojoModel
   .forward` pass. The staged ``representations`` /
   ``predict_from_representations`` / KV-cache API used by the Torch and NumPy
   backends is not implemented yet; those methods raise
   :class:`NotImplementedError` until the kernels reach numeric parity.

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

    def forward(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Run the native end-to-end forward pass.

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
        n_train = int(np.asarray(y_train).shape[0])
        n_test = x.shape[0] - n_train

        group_size = cfg.col_feature_group_size
        col_input = np.ascontiguousarray(x[:n_train, :group_size], dtype=np.float32)
        row_input = np.zeros((max(n_test, 1), cfg.embed_dim), dtype=np.float32)
        target = np.asarray(y_train, dtype=np.int64)

        out = np.asarray(
            self._handle.forward([col_input, row_input, target]), dtype=np.float32
        )
        if n_test:
            out = out[: n_test * cfg.out_dim].reshape(n_test, cfg.out_dim)
        else:
            out = out[:0]

        if cfg.max_classes > 0 and not return_logits and out.size:
            out = np.exp(out) / np.sum(np.exp(out), axis=-1, keepdims=True)
        return out

    # ------------------------------------------------------------------ #
    # staged API (pending native parity work)
    # ------------------------------------------------------------------ #

    def representations(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Column embedding + row interaction (not implemented natively yet)."""
        raise NotImplementedError(
            "The Mojo backend does not implement the staged inference API yet; "
            "use TabICLMojoModel.forward() or the 'numpy'/'torch' backends."
        )

    def predict_from_representations(
        self,
        R: np.ndarray,
        y_train: np.ndarray,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """ICL-stage decoding (not implemented natively yet)."""
        raise NotImplementedError(
            "The Mojo backend does not implement the staged inference API yet; "
            "use TabICLMojoModel.forward() or the 'numpy'/'torch' backends."
        )

    def build_cache(self, X: np.ndarray, y_train: np.ndarray) -> dict:
        """KV-cache construction (not implemented natively yet)."""
        raise NotImplementedError(
            "The Mojo backend does not support KV caching yet; "
            "use the 'numpy'/'torch' backends."
        )

    def predict_with_cache(
        self,
        X_test: np.ndarray,
        cache: dict,
        return_logits: bool = True,
        temperature: float = 0.9,
    ) -> np.ndarray:
        """Cached prediction (not implemented natively yet)."""
        raise NotImplementedError(
            "The Mojo backend does not support KV caching yet; "
            "use the 'numpy'/'torch' backends."
        )

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
