"""Backend-agnostic KV cache structures for TabICL inference.

TabICL uses KV caching at two stages to avoid recomputing attention keys/values
for the training data on every prediction:

1. **ColEmbedding (SetTransformer)** — K/V of the second attention layer
   (``multihead_attn2``) of each ISAB block, projected from training rows.
2. **ICLearning (Encoder)** — K/V from training rows at each encoder layer,
   after target embedding is added to training positions.

The cache is built once during ``fit()`` (when ``kv_cache=True``) and reused
across all ``predict()`` calls. Each entry is a pair of ``(key, value)``
arrays with shapes ``(..., num_heads, seq_len, head_dim)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TabICLCache:
    """Container for TabICL KV caches across both stages.

    Attributes:
        col: List of ``(key, value)`` K/V pairs per ISAB block in the
            column embedding stage. Length equals ``cfg.col_num_blocks``.
        icl: List of ``(key, value)`` K/V pairs per encoder layer in the
            ICL stage. Length equals ``cfg.icl_num_blocks``.
        train_size: Number of training rows (used to slice training vs test).
        num_classes: Number of unique classes (0 for regression).
        _backend: Backend name that built this cache (for validation).
    """

    col: list[tuple[Any, Any]] = field(default_factory=list)
    icl: list[tuple[Any, Any]] = field(default_factory=list)
    train_size: int = 0
    num_classes: int = 0
    _backend: str = ""

    def validate(self, col_blocks: int, icl_blocks: int, backend: str) -> None:
        """Validate cache is consistent with the model configuration."""
        if backend != self._backend and self._backend:
            raise ValueError(
                f"Cache built with backend '{self._backend}', "
                f"but model uses '{backend}'"
            )
        if len(self.col) != col_blocks:
            raise ValueError(
                f"Expected {col_blocks} col cache entries, got {len(self.col)}"
            )
        if len(self.icl) != icl_blocks:
            raise ValueError(
                f"Expected {icl_blocks} icl cache entries, got {len(self.icl)}"
            )
        if self.train_size <= 0:
            raise ValueError(f"Invalid train_size: {self.train_size}")

    @classmethod
    def empty(cls, backend: str) -> TabICLCache:
        """Create an empty cache for a given backend."""
        return cls(col=[], icl=[], train_size=0, num_classes=0, _backend=backend)

    def is_empty(self) -> bool:
        """Return True when no cache entries have been populated."""
        return len(self.col) == 0 and len(self.icl) == 0
