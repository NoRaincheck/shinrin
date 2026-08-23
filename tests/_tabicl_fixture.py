"""Shared helpers for TabICL tests: tiny synthetic checkpoints (no network).

Builds minimal TabICLv2 weights under *upstream torch state-dict naming* so
that every backend (torch strict loader, NumPy resolver) consumes the same
arrays, then packages them as a converted ``.npz`` archive plus a placeholder
``.ckpt`` file so ``ensure_npz`` never touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from shinrin._tabicl._checkpoint import NPZ_FORMAT_VERSION

TINY_CLASSIFIER: dict[str, Any] = {
    "max_classes": 4,
    "num_quantiles": 16,
    "embed_dim": 16,
    "col_num_blocks": 1,
    "col_nhead": 2,
    "col_num_inds": 4,
    "col_affine": False,
    "col_feature_group": "same",
    "col_feature_group_size": 3,
    "col_target_aware": True,
    "col_ssmax": "qassmax-mlp-elementwise",
    "row_num_blocks": 2,
    "row_nhead": 2,
    "row_num_cls": 2,
    "row_rope_base": 100000.0,
    "row_rope_interleaved": True,
    "icl_num_blocks": 2,
    "icl_nhead": 2,
    "icl_ssmax": "qassmax-mlp-elementwise",
    "ff_factor": 2,
    "dropout": 0.0,
    "activation": "gelu",
    "norm_first": True,
    "bias_free_ln": False,
    "zero_init": True,
    "recompute": False,
}

TINY_REGRESSOR: dict[str, Any] = {
    **TINY_CLASSIFIER,
    "max_classes": 0,
}


def _linear(P, prefix, out_f, in_f, rng, scale=0.05):
    P[f"{prefix}.weight"] = (rng.randn(out_f, in_f) * scale).astype(np.float32)
    P[f"{prefix}.bias"] = (rng.randn(out_f) * scale).astype(np.float32)


def _attention_block(P, prefix, d_model, d_ff, nhead, head_dim, ssmax, rng):
    """Emit tensors for one torch AttentionBlock (``prefix`` ends at .attn)."""
    P[f"{prefix}.in_proj_weight"] = (rng.randn(3 * d_model, d_model) * 0.05).astype(
        np.float32
    )
    P[f"{prefix}.in_proj_bias"] = (rng.randn(3 * d_model) * 0.05).astype(np.float32)
    _linear(P, f"{prefix}.out_proj", d_model, d_model, rng)
    if ssmax:
        s = f"{prefix}.ssmax_layer"
        hidden = 64
        P[f"{s}.base_mlp.0.weight"] = (rng.randn(hidden, 1) * 0.1).astype(np.float32)
        P[f"{s}.base_mlp.0.bias"] = np.zeros(hidden, dtype=np.float32)
        P[f"{s}.base_mlp.2.weight"] = (
            rng.randn(nhead * head_dim, hidden) * 0.1
        ).astype(np.float32)
        P[f"{s}.base_mlp.2.bias"] = (rng.randn(nhead * head_dim) * 0.1).astype(
            np.float32
        )
        P[f"{s}.query_mlp.0.weight"] = (rng.randn(hidden, head_dim) * 0.1).astype(
            np.float32
        )
        P[f"{s}.query_mlp.0.bias"] = np.zeros(hidden, dtype=np.float32)
        P[f"{s}.query_mlp.2.weight"] = (rng.randn(head_dim, hidden) * 0.1).astype(
            np.float32
        )
        P[f"{s}.query_mlp.2.bias"] = (rng.randn(head_dim) * 0.1).astype(np.float32)


def make_params(cfg: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Random upstream-style state dict for a tiny TabICLv2 config."""
    rng = np.random.RandomState(seed)
    cfg = dict(cfg)
    cfg.setdefault("out_dim", cfg["max_classes"] or cfg["num_quantiles"])
    P: dict[str, np.ndarray] = {}
    e = cfg["embed_dim"]
    cff = e * cfg["ff_factor"]
    icl_d = e * cfg["row_num_cls"]
    iffl = icl_d * cfg["ff_factor"]
    chd = e // cfg["col_nhead"]

    def _block_extras(pfx, d_model, d_ff):
        """FFN/LayerNorm tensors living directly on torch AttentionBlock."""
        _linear(P, f"{pfx}.linear1", d_ff, d_model, rng)
        _linear(P, f"{pfx}.linear2", d_model, d_ff, rng)
        for n in (1, 2):
            P[f"{pfx}.norm{n}.weight"] = np.ones(d_model, dtype=np.float32)
            P[f"{pfx}.norm{n}.bias"] = (rng.randn(d_model) * 0.05).astype(np.float32)

    # Stage 1: ColEmbedding
    _linear(P, "col_embedder.in_linear", e, cfg["col_feature_group_size"], rng)
    if cfg["max_classes"] > 0:
        _linear(P, "col_embedder.y_encoder", e, cfg["max_classes"], rng)
    else:
        _linear(P, "col_embedder.y_encoder", e, 1, rng)
    for b in range(cfg["col_num_blocks"]):
        p = f"col_embedder.tf_col.blocks.{b}"
        P[f"{p}.ind_vectors"] = (rng.randn(cfg["col_num_inds"], e) * 0.05).astype(
            np.float32
        )
        _attention_block(
            P,
            f"{p}.multihead_attn1.attn",
            e,
            cff,
            cfg["col_nhead"],
            chd,
            True,
            rng,
        )
        _block_extras(f"{p}.multihead_attn1", e, cff)
        _attention_block(
            P,
            f"{p}.multihead_attn2.attn",
            e,
            cff,
            cfg["col_nhead"],
            chd,
            False,
            rng,
        )
        _block_extras(f"{p}.multihead_attn2", e, cff)

    # Stage 2: RowInteraction
    P["row_interactor.cls_tokens"] = (rng.randn(cfg["row_num_cls"], e) * 0.05).astype(
        np.float32
    )
    P["row_interactor.out_ln.weight"] = np.ones(e, dtype=np.float32)
    P["row_interactor.out_ln.bias"] = (rng.randn(e) * 0.05).astype(np.float32)
    dim = max(e // cfg["row_nhead"], 2)
    freqs = (
        1.0
        / (
            float(cfg["row_rope_base"])
            ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
    ).astype(np.float32)
    P["row_interactor.tf_row.rope.freqs"] = freqs
    for b in range(cfg["row_num_blocks"]):
        p = f"row_interactor.tf_row.blocks.{b}"
        _attention_block(
            P,
            f"{p}.attn",
            e,
            cff,
            cfg["row_nhead"],
            chd,
            False,
            rng,
        )
        _block_extras(p, e, cff)

    # Stage 3: ICLearning
    ihd = icl_d // cfg["icl_nhead"]
    P["icl_predictor.ln.weight"] = np.ones(icl_d, dtype=np.float32)
    P["icl_predictor.ln.bias"] = (rng.randn(icl_d) * 0.05).astype(np.float32)
    if cfg["max_classes"] > 0:
        _linear(P, "icl_predictor.y_encoder", icl_d, cfg["max_classes"], rng)
    else:
        _linear(P, "icl_predictor.y_encoder", icl_d, 1, rng)
    _linear(P, "icl_predictor.decoder.0", icl_d * 2, icl_d, rng)
    _linear(P, "icl_predictor.decoder.2", cfg["out_dim"] or 16, icl_d * 2, rng)
    for b in range(cfg["icl_num_blocks"]):
        p = f"icl_predictor.tf_icl.blocks.{b}"
        _attention_block(
            P,
            f"{p}.attn",
            icl_d,
            iffl,
            cfg["icl_nhead"],
            ihd,
            bool(cfg["icl_ssmax"]),
            rng,
        )
        _block_extras(p, icl_d, iffl)
    return P


def write_synthetic_checkpoint(
    directory: Path,
    stem: str,
    cfg: dict | None = None,
    seed: int = 0,
) -> Path:
    """Create ``<stem>.ckpt`` (placeholder) + ``<stem>.npz`` in ``directory``.

    Returns the placeholder checkpoint path that ``model_path`` should point
    at (the estimators derive the ``.npz`` sibling from it).
    """
    directory.mkdir(parents=True, exist_ok=True)
    cfg = dict(TINY_CLASSIFIER if cfg is None else cfg)
    if cfg["max_classes"] == 0:
        cfg["out_dim"] = cfg["num_quantiles"]
    else:
        cfg["out_dim"] = cfg["max_classes"]
    params = make_params(cfg, seed=seed)
    npz_payload: dict[str, ArrayLike] = {
        "__format_version__": np.array(NPZ_FORMAT_VERSION),
        "__config__": np.array(json.dumps(cfg)),
    }
    for name, arr in params.items():
        npz_payload[name] = np.asarray(arr, dtype=np.float32)
    npz_path = directory / f"{stem}.npz"
    with open(npz_path, "wb") as fh:
        # ty: ignore[invalid-argument-type] - numpy stub misreads **kwargs
        np.savez(fh, **npz_payload)
    ckpt_path = directory / f"{stem}.ckpt"
    ckpt_path.write_bytes(b"synthetic-checkpoint-placeholder")
    return ckpt_path
