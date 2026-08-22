"""TabICL model configuration parsed from checkpoint metadata.

The Hugging Face checkpoints (``jingang/TabICL``) store the constructor
keyword arguments of the upstream ``TabICL`` module under the ``config``
key. We parse them into a frozen dataclass with the upstream defaults so
that all backends (torch, NumPy, Mojo) share one source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TabICLConfig:
    """Architecture hyper-parameters of a TabICL checkpoint."""

    max_classes: int = 10
    num_quantiles: int = 999
    embed_dim: int = 128
    col_num_blocks: int = 3
    col_nhead: int = 8
    col_num_inds: int = 128
    col_affine: bool = False
    col_feature_group: Any = "same"
    col_feature_group_size: int = 3
    col_target_aware: bool = True
    col_ssmax: str = "qassmax-mlp-elementwise"
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    row_rope_base: float = 100000.0
    row_rope_interleaved: bool = True
    icl_num_blocks: int = 12
    icl_nhead: int = 8
    icl_ssmax: str = "qassmax-mlp-elementwise"
    ff_factor: int = 2
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    bias_free_ln: bool = False
    zero_init: bool = True
    recompute: bool = False
    # Hidden width of the SSMax scale MLPs (upstream default).
    col_ssmax_n_hidden: int = 64
    icl_ssmax_n_hidden: int = 64

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TabICLConfig:
        """Build a config from checkpoint metadata, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "col_feature_group" in kwargs and kwargs["col_feature_group"] is True:
            kwargs["col_feature_group"] = "same"
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict."""
        return asdict(self)

    @staticmethod
    def _ssmax_flags(ssmax: str) -> tuple[int, int]:
        """Map an SSMax variant name to ``(elementwise, n_hidden)`` flags.

        The Mojo kernels encode the SSMax configuration numerically; only
        ``*mlp`` variants carry a scale MLP (hidden width 64 upstream), and
        only ``*-elementwise`` variants emit per-dimension scales.
        """
        name = str(ssmax).lower()
        elementwise = 1 if "elementwise" in name else 0
        hidden = 64 if "mlp" in name else 0
        return elementwise, hidden

    def dims_array(self) -> np.ndarray:
        """Pack the architecture hyper-parameters for the Mojo kernels.

        The field order is fixed by ``TabICLConfig`` in
        ``_tabicl_kernels.mojo``; keep both sides in sync. ``row_rope_base``
        is stored as an integer (exact for the upstream value 100000.0) and
        converted back to float on the native side.
        """
        col_elementwise, _ = self._ssmax_flags(self.col_ssmax)
        icl_elementwise, _ = self._ssmax_flags(self.icl_ssmax)
        return np.array(
            [
                self.embed_dim,
                self.col_feature_group_size,
                self.col_num_blocks,
                self.col_nhead,
                self.col_num_inds,
                1 if self.col_target_aware else 0,
                col_elementwise,
                self.col_ssmax_n_hidden if "mlp" in str(self.col_ssmax).lower() else 0,
                self.row_num_cls,
                self.row_num_blocks,
                self.row_nhead,
                round(float(self.row_rope_base)),
                1 if self.row_rope_interleaved else 0,
                self.icl_num_blocks,
                self.icl_nhead,
                self.icl_dim,
                icl_elementwise,
                self.icl_ssmax_n_hidden if "mlp" in str(self.icl_ssmax).lower() else 0,
                self.ff_factor,
                1 if self.bias_free_ln else 0,
                self.max_classes,
                self.num_quantiles,
                self.out_dim,
            ],
            dtype=np.int64,
        )

    @property
    def is_regression(self) -> bool:
        """True when the checkpoint is a regressor (``max_classes == 0``)."""
        return self.max_classes == 0

    @property
    def icl_dim(self) -> int:
        """Dimension of row representations (CLS tokens concatenated)."""
        return self.embed_dim * self.row_num_cls

    @property
    def out_dim(self) -> int:
        """Dimension of the ICL decoder output."""
        if self.is_regression:
            return self.num_quantiles
        return self.max_classes

    @property
    def col_dim_feedforward(self) -> int:
        """Feedforward width of the column set transformer."""
        return self.embed_dim * self.ff_factor

    @property
    def icl_dim_feedforward(self) -> int:
        """Feedforward width of the ICL transformer."""
        return self.icl_dim * self.ff_factor
