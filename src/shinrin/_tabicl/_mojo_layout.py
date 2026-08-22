"""Canonical flat parameter layout for the Mojo TabICL kernels.

This module is the **single source of truth** for how a TabICLv2 state dict
is flattened into the one-dimensional float32 buffer consumed by
``shinrin._native_tabicl``. The kernel's internal offset walk in
``_tabicl_kernels.mojo`` mirrors :func:`canonical_tensors` exactly; the
kernel additionally validates at construction time that the packed buffer
length equals its own walk total, so any future divergence between the two
sides fails loudly instead of silently reading misaligned weights.

The tensor names and shapes follow the upstream torch state-dict naming
(the same names produced by ``tests/_tabicl_fixture.make_params`` and the
converted ``.npz`` checkpoints).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = [
    "SSMAX_KINDS",
    "canonical_tensor_specs",
    "canonical_tensors",
    "pack_params",
    "ssmax_kind",
    "total_elems",
]

#: SSMax variant identifiers shared with the Mojo kernels (numeric values
#: are part of the ``dims_array`` contract; do not renumber).
SSMAX_NONE = 0
SSMAX_SCALES = 1  # "ssmax": learnable per-head scales
SSMAX_MLP = 2  # "ssmax-mlp"
SSMAX_MLP_ELEMENTWISE = 3  # "ssmax-mlp-elementwise"
SSMAX_QAMLP = 4  # "qassmax-mlp"
SSMAX_QAMLP_ELEMENTWISE = 5  # "qassmax-mlp-elementwise"

SSMAX_KINDS = {
    "none": SSMAX_NONE,
    "ssmax": SSMAX_SCALES,
    "ssmax-mlp": SSMAX_MLP,
    "ssmax-mlp-elementwise": SSMAX_MLP_ELEMENTWISE,
    "qassmax-mlp": SSMAX_QAMLP,
    "qassmax-mlp-elementwise": SSMAX_QAMLP_ELEMENTWISE,
}


def ssmax_kind(name: str | bool | None) -> int:
    """Map an SSMax variant name to its numeric kernel identifier."""
    if name in (None, False, "none"):
        return SSMAX_NONE
    if name is True:
        name = "qassmax-mlp-elementwise"
    key = str(name).strip().lower()
    if key not in SSMAX_KINDS:
        raise ValueError(f"Unknown ssmax variant: {name!r}")
    return SSMAX_KINDS[key]


TensorSpec = tuple[str, tuple[int, ...]]


class _SpecBuilder:
    """Accumulates ``(name, shape)`` pairs in the canonical flat order."""

    def __init__(self, bias_free_ln: bool) -> None:
        self.specs: list[TensorSpec] = []
        self._bias_free = bias_free_ln

    def linear(self, prefix: str, out_features: int, in_features: int) -> None:
        self.specs.append((f"{prefix}.weight", (out_features, in_features)))
        self.specs.append((f"{prefix}.bias", (out_features,)))

    def layernorm(self, prefix: str, width: int) -> None:
        self.specs.append((f"{prefix}.weight", (width,)))
        if not self._bias_free:
            self.specs.append((f"{prefix}.bias", (width,)))

    def ssmax(
        self, prefix: str, kind: int, n_heads: int, head_dim: int, hidden: int
    ) -> None:
        if kind == SSMAX_NONE:
            return
        if kind == SSMAX_SCALES:
            self.specs.append((f"{prefix}.scales", (n_heads,)))
            return
        elementwise = kind in (SSMAX_MLP_ELEMENTWISE, SSMAX_QAMLP_ELEMENTWISE)
        base_out = n_heads * head_dim if elementwise else n_heads
        # SSMaxMLP wraps a single ``mlp``; QASSMaxMLP splits base/query MLPs.
        base_prefix = (
            "base_mlp" if kind in (SSMAX_QAMLP, SSMAX_QAMLP_ELEMENTWISE) else "mlp"
        )
        self.specs.append((f"{prefix}.{base_prefix}.0.weight", (hidden, 1)))
        self.specs.append((f"{prefix}.{base_prefix}.0.bias", (hidden,)))
        self.specs.append((f"{prefix}.{base_prefix}.2.weight", (base_out, hidden)))
        self.specs.append((f"{prefix}.{base_prefix}.2.bias", (base_out,)))
        if kind in (SSMAX_QAMLP, SSMAX_QAMLP_ELEMENTWISE):
            query_out = head_dim if elementwise else 1
            self.specs.append((f"{prefix}.query_mlp.0.weight", (hidden, head_dim)))
            self.specs.append((f"{prefix}.query_mlp.0.bias", (hidden,)))
            self.specs.append((f"{prefix}.query_mlp.2.weight", (query_out, hidden)))
            self.specs.append((f"{prefix}.query_mlp.2.bias", (query_out,)))

    def attention_block(
        self,
        prefix: str,
        d_model: int,
        d_ff: int,
        n_heads: int,
        head_dim: int,
        kind: int,
        hidden: int,
    ) -> None:
        """One torch ``AttentionBlock`` living at ``prefix``.

        ``prefix.attn`` is the ``nn.MultiheadAttention`` submodule (in/out
        projections and the ssmax layer); norm1/norm2/linear1/linear2 are
        direct children of the block itself.
        """
        a = f"{prefix}.attn"
        self.specs.append((f"{a}.in_proj_weight", (3 * d_model, d_model)))
        self.specs.append((f"{a}.in_proj_bias", (3 * d_model,)))
        self.linear(f"{a}.out_proj", d_model, d_model)
        self.ssmax(f"{a}.ssmax_layer", kind, n_heads, head_dim, hidden)
        self.layernorm(f"{prefix}.norm1", d_model)
        self.layernorm(f"{prefix}.norm2", d_model)
        self.linear(f"{prefix}.linear1", d_ff, d_model)
        self.linear(f"{prefix}.linear2", d_model, d_ff)


def canonical_tensor_specs(cfg) -> list[TensorSpec]:
    """Return the ordered ``(name, shape)`` specification for ``cfg``.

    The order defines the flat parameter buffer layout and must stay in
    sync with the offset walk in ``TabICLInference.__init__``
    (``_tabicl_kernels.mojo``).
    """
    e = cfg.embed_dim
    cff = cfg.col_dim_feedforward
    chd = e // cfg.col_nhead
    icl_d = cfg.icl_dim
    iffl = cfg.icl_dim_feedforward
    ihd = icl_d // cfg.icl_nhead
    rhd = e // cfg.row_nhead
    n_cls = cfg.max_classes
    out_d = cfg.out_dim
    col_kind = ssmax_kind(cfg.col_ssmax)
    icl_kind = ssmax_kind(cfg.icl_ssmax)
    b = _SpecBuilder(cfg.bias_free_ln)

    # -- Stage 1: ColEmbedding ------------------------------------------- #
    b.linear("col_embedder.in_linear", e, cfg.col_feature_group_size)
    if cfg.col_target_aware:
        b.linear("col_embedder.y_encoder", e, n_cls if n_cls > 0 else 1)
    if getattr(cfg, "col_affine", False):
        b.linear("col_embedder.out_w", e, e)
        b.linear("col_embedder.out_b", e, e)
        b.layernorm("col_embedder.ln_w", e)
        b.layernorm("col_embedder.ln_b", e)
    for i in range(cfg.col_num_blocks):
        p = f"col_embedder.tf_col.blocks.{i}"
        b.specs.append((f"{p}.ind_vectors", (cfg.col_num_inds, e)))
        b.attention_block(
            f"{p}.multihead_attn1",
            e,
            cff,
            cfg.col_nhead,
            chd,
            col_kind,
            cfg.col_ssmax_n_hidden,
        )
        b.attention_block(
            f"{p}.multihead_attn2",
            e,
            cff,
            cfg.col_nhead,
            chd,
            SSMAX_NONE,
            cfg.col_ssmax_n_hidden,
        )

    # -- Stage 2: RowInteraction ----------------------------------------- #
    b.specs.append(("row_interactor.cls_tokens", (cfg.row_num_cls, e)))
    b.layernorm("row_interactor.out_ln", e)
    b.specs.append(("row_interactor.tf_row.rope.freqs", ((rhd + 1) // 2,)))
    for i in range(cfg.row_num_blocks):
        b.attention_block(
            f"row_interactor.tf_row.blocks.{i}",
            e,
            cff,
            cfg.row_nhead,
            rhd,
            SSMAX_NONE,
            cfg.col_ssmax_n_hidden,
        )

    # -- Stage 3: ICLearning ---------------------------------------------- #
    b.layernorm("icl_predictor.ln", icl_d)
    b.linear("icl_predictor.y_encoder", icl_d, n_cls if n_cls > 0 else 1)
    b.linear("icl_predictor.decoder.0", icl_d * 2, icl_d)
    b.linear("icl_predictor.decoder.2", out_d, icl_d * 2)
    for i in range(cfg.icl_num_blocks):
        b.attention_block(
            f"icl_predictor.tf_icl.blocks.{i}",
            icl_d,
            iffl,
            cfg.icl_nhead,
            ihd,
            icl_kind,
            cfg.icl_ssmax_n_hidden,
        )
    return b.specs


def canonical_tensors(cfg) -> dict[str, tuple[int, ...]]:
    """Mapping form of :func:`canonical_tensor_specs` (name -> shape)."""
    return dict(canonical_tensor_specs(cfg))


def total_elems(cfg) -> int:
    """Number of float32 values the canonical layout requires for ``cfg``."""
    total = 0
    for _, shape in canonical_tensor_specs(cfg):
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


def iter_flat(params: dict[str, np.ndarray]) -> Iterator[np.ndarray]:
    for arr in params.values():
        yield np.asarray(arr, dtype=np.float32).ravel()


def pack_params(cfg, params: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten ``params`` into the canonical buffer order.

    Raises :class:`KeyError` / :class:`ValueError` when the state dict does
    not exactly cover the canonical layout (missing tensors, unexpected
    tensors, or shape mismatches) — the kernels fail fast on the same
    conditions from the native side, so a mismatch can never reach the
    numeric path silently.
    """
    parts: list[np.ndarray] = []
    remaining = dict(params)
    for name, shape in canonical_tensor_specs(cfg):
        if name not in params:
            raise KeyError(f"state dict is missing layout tensor {name!r}")
        arr = np.asarray(params[name], dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(
                f"tensor {name!r} has shape {arr.shape}, layout expects {shape}"
            )
        del remaining[name]
        parts.append(np.ascontiguousarray(arr).ravel())
    if remaining:
        extra = ", ".join(sorted(remaining)[:8])
        raise ValueError(
            f"state dict has {len(remaining)} tensor(s) unknown to the layout "
            f"(first few: {extra})"
        )
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
