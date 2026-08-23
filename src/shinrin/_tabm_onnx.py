"""ONNX export for the vendored TabM estimators.

Converts a fitted :class:`~shinrin.tabm.TabMClassifier` /
:class:`~shinrin.tabm.TabMRegressor` into a self-contained ONNX graph.
The exported graph reproduces the full inference pipeline — preprocessing
(quantile / asinh / standard-scaler transforms, piecewise-linear encoding,
categorical one-hot), the optional PLE embedding layer, the BatchEnsemble
backbone and the ensemble-averaged head — so deployment needs nothing but
the ``.onnx`` file and raw feature vectors.

Design notes
------------
- Only standard-domain ops are emitted (default opset 15, matching
  :mod:`shinrin.onnx`); the ``ai.onnx.ml`` domain is not used.
- ``Einsum`` is avoided in favour of ``Transpose``/``MatMul`` pairs for
  maximum runtime portability.
- Per-feature piecewise-linear encodings with differing bin counts are
  expressed with a single padded computation plus validity masks, keeping
  the graph size independent of the number of features.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

__all__ = ["tabm_to_onnx"]

DEFAULT_OPSET = 15


class _GraphBuilder:
    """Small helper that collects nodes and initializers for a graph."""

    def __init__(self) -> None:
        self.nodes: list = []
        self.initializers: list = []
        self._counter: dict[str, int] = {}

    def fresh(self, prefix: str) -> str:
        n = self._counter.get(prefix, 0)
        self._counter[prefix] = n + 1
        return f"{prefix}_{n}" if n else prefix

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def op(self, op_type: str, inputs: list[str], output: str | None = None, **attrs):
        name = output or self.fresh(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [name], **attrs))
        return name

    def konst(self, name: str, array: np.ndarray) -> str:
        return self.init(name, np.asarray(array))


def _f32(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing subgraphs
# ---------------------------------------------------------------------------


def _gather_numeric_columns(
    gb: _GraphBuilder,
    x: str,
    num_indices: list[int],
    n_features_in: int,
) -> str:
    if len(num_indices) == n_features_in and num_indices == list(range(n_features_in)):
        return x
    idx = gb.init("num_col_idx", np.array(num_indices, dtype=np.int64))
    return gb.op("Gather", [x, idx], axis=1)


def _apply_numeric_transforms(
    gb: _GraphBuilder,
    pre,
    x: str,
    f_idx: int,
) -> str:
    """Apply the fitted quantile/asinh/scaler transforms in fitted order."""
    for t in pre.transforms_:
        kind = type(t).__name__
        if kind == "QuantileTransform":
            bounds = gb.init(f"quant_bounds_{f_idx}", _f32(t.boundaries_))
            denom = gb.konst(
                f"quant_denom_{f_idx}",
                np.array(t.num_quantiles, dtype=np.float32),
            )
            xu = gb.op(
                "Unsqueeze", [x, gb.init(f"ax2_{f_idx}", np.array([2], dtype=np.int64))]
            )
            bounds_u = gb.op(
                "Unsqueeze",
                [bounds, gb.init(f"ax0_{f_idx}", np.array([0], dtype=np.int64))],
            )
            ge = gb.op("GreaterOrEqual", [xu, bounds_u])
            counts = gb.op(
                "ReduceSum",
                [
                    gb.op("Cast", [ge], to=TensorProto.FLOAT),
                    gb.init(f"red_ax_{f_idx}", np.array([2], dtype=np.int64)),
                ],
                keepdims=0,
            )
            x = gb.op("Div", [counts, denom])
        elif kind == "AsinhTransform":
            x = gb.op("Asinh", [x])
        elif kind == "StandardScalerTransform":
            mean = gb.init(f"scaler_mean_{f_idx}", _f32(t.mean_))
            std = gb.init(f"scaler_std_{f_idx}", _f32(t.std_))
            centered = gb.op("Sub", [x, mean])
            x = gb.op("Div", [centered, std])
        else:  # pragma: no cover
            raise ValueError(f"Unsupported TabM transform: {kind}")
    return x


def _add_piecewise_linear_embedding(
    gb: _GraphBuilder,
    cfg,
    pre,
    params_arrays: dict,
    x_num_raw: str,
    x_num_t: str,
    f_idx: int,
) -> str:
    """Build the PLE + linear embedding; returns the flat embedding tensor.

    Mirrors ``TabMCore._embed``: the *linear* term consumes the raw
    numerical features while the piecewise-linear branch consumes the
    *transformed* features (quantile/asinh/scaler outputs), matching
    ``_Preprocessor.transform`` which returns raw ``x_num`` next to the
    encoding of its transformed copy.
    """
    bins = pre.bins_
    n_feat = len(bins)
    demb = cfg.d_embedding
    max_bins = max(len(edges) - 1 for edges in bins)

    # Padded per-feature tables. Positions beyond a feature's own bin count
    # get width 1.0 and rule id 0 (invalid), so their components are masked
    # to zero and their weight rows are zero.
    edges_lo = np.zeros((n_feat, max_bins), dtype=np.float32)
    widths = np.ones((n_feat, max_bins), dtype=np.float32)
    rules = np.zeros((n_feat, max_bins), dtype=np.int64)  # 0 invalid
    wp_all = np.zeros((n_feat, max_bins, demb), dtype=np.float32)
    for f, edges in enumerate(bins):
        m = len(edges) - 1
        diffs = np.diff(edges).astype(np.float32)
        edges_lo[f, :m] = edges[:-1]
        widths[f, :m] = diffs
        if m == 1:
            rules[f, 0] = 4  # single-bin: unclamped min-max scaling
        else:
            rules[f, 0] = 1  # first component: clamped above at 1
            rules[f, 1 : m - 1] = 2  # interior: clipped to [0, 1]
            rules[f, m - 1] = 3  # last component: clamped below at 0
        wp_all[f, :m] = params_arrays[f"emb_wp_{f}"]

    edges_i = gb.init(f"ple_edges_lo_{f_idx}", edges_lo)
    widths_i = gb.init(f"ple_widths_{f_idx}", widths)
    rules_i = gb.init(f"ple_rules_{f_idx}", rules)

    x3 = gb.op(
        "Unsqueeze",
        [x_num_t, gb.init(f"ple_ax2_{f_idx}", np.array([2], dtype=np.int64))],
    )  # (B, F, 1)
    t = gb.op("Div", [gb.op("Sub", [x3, edges_i]), widths_i])  # (B, F, E)

    one = gb.konst("ple_one", np.array(1.0, dtype=np.float32))
    zero = gb.konst("ple_zero", np.array(0.0, dtype=np.float32))
    first_out = gb.op("Min", [t, one])
    interior_out = gb.op("Clip", [t, zero, one])
    last_out = gb.op("Max", [t, zero])

    r_eq1 = gb.op("Equal", [rules_i, gb.konst("ple_r1", np.array(1, dtype=np.int64))])
    r_eq2 = gb.op("Equal", [rules_i, gb.konst("ple_r2", np.array(2, dtype=np.int64))])
    r_eq3 = gb.op("Equal", [rules_i, gb.konst("ple_r3", np.array(3, dtype=np.int64))])
    sel1 = gb.op("Where", [r_eq1, first_out, t])
    sel2 = gb.op("Where", [r_eq2, interior_out, sel1])
    sel3 = gb.op("Where", [r_eq3, last_out, sel2])
    valid = gb.op(
        "Cast",
        [
            gb.op(
                "GreaterOrEqual",
                [rules_i, gb.konst("ple_r0", np.array(1, dtype=np.int64))],
            )
        ],
        to=TensorProto.FLOAT,
    )
    enc = gb.op("Mul", [sel3, valid])  # (B, F, E)

    # pl[b, f, :] = sum_j enc[b, f, j] * wp_all[f, j, :]
    enc_t = gb.op("Transpose", [enc], perm=[1, 0, 2])  # (F, B, E)
    pl_t = gb.op("MatMul", [enc_t, gb.init(f"ple_wp_{f_idx}", wp_all)])  # (F, B, demb)
    pl = gb.op("Transpose", [pl_t], perm=[1, 0, 2])  # (B, F, demb)

    w0 = gb.init(f"emb_w0_{f_idx}", _f32(params_arrays["emb_w0"]))
    b0 = gb.init(f"emb_b0_{f_idx}", _f32(params_arrays["emb_b0"]))
    raw3 = gb.op(
        "Unsqueeze",
        [x_num_raw, gb.init(f"raw_ax2_{f_idx}", np.array([2], dtype=np.int64))],
    )  # (B, F, 1)
    lin0 = gb.op("Add", [gb.op("Mul", [raw3, w0]), b0])  # (B, F, demb)
    emb = gb.op("Add", [lin0, gb.op("Relu", [pl])])

    flat_shape = gb.init(
        f"emb_shape_{f_idx}", np.array([-1, n_feat * demb], dtype=np.int64)
    )
    flat = gb.op("Reshape", [emb, flat_shape])
    return flat


def _add_categorical_onehots(
    gb: _GraphBuilder,
    pre,
    x: str,
    tag: str,
) -> list[str]:
    parts: list[str] = []
    for pos, (column, mapping) in enumerate(
        zip(pre.categorical_indices_, pre.value_maps_)
    ):
        values = np.asarray(list(mapping.keys()), dtype=np.float32)
        card = len(values)
        col = gb.op(
            "Slice",
            [
                x,
                gb.init(f"{tag}_start_{pos}", np.array([column], dtype=np.int64)),
                gb.init(f"{tag}_end_{pos}", np.array([column + 1], dtype=np.int64)),
                gb.init(f"{tag}_ax_{pos}", np.array([1], dtype=np.int64)),
            ],
        )  # (B, 1)
        vals = gb.init(f"{tag}_vals_{pos}", values.reshape(1, card))
        le = gb.op("LessOrEqual", [vals, col])  # (B, C)
        counts = gb.op(
            "ReduceSum",
            [
                gb.op("Cast", [le], to=TensorProto.FLOAT),
                gb.init(f"{tag}_rax_{pos}", np.array([1], dtype=np.int64)),
            ],
            keepdims=0,
        )  # (B,))
        one_f = gb.konst(f"{tag}_one_{pos}", np.array(1.0, dtype=np.float32))
        zero = gb.konst(f"{tag}_zero_{pos}", np.array(0.0, dtype=np.float32))
        # searchsorted-right minus one gives the matching bucket index;
        # unseen values below the minimum clamp to 0.
        idx_f = gb.op("Max", [gb.op("Sub", [counts, one_f]), zero])
        idx = gb.op("Cast", [idx_f], to=TensorProto.INT64)
        onehot = gb.op(
            "OneHot",
            [
                idx,
                gb.konst(f"{tag}_depth_{pos}", np.array(card, dtype=np.int64)),
                gb.konst(f"{tag}_ohv_{pos}", np.array([0.0, 1.0], dtype=np.float32)),
            ],
            axis=1,
        )
        parts.append(onehot)
    return parts


# ---------------------------------------------------------------------------
# Backbone and head
# ---------------------------------------------------------------------------


def _add_tabm_block(gb: _GraphBuilder, cfg, arrays: dict, i: int, x: str) -> str:
    """One BatchEnsemble block: ``(x * r) @ W.T * s + b`` then ReLU."""
    prefix = f"blk{i}_"
    w = gb.init(f"{prefix}w", _f32(arrays[prefix + "w"]))
    wt = gb.op("Transpose", [w], perm=[1, 0])  # (d_in_i, d_block)
    r = gb.init(f"{prefix}r", _f32(arrays[prefix + "r"]))
    r_u = gb.op(
        "Unsqueeze", [r, gb.init(f"{prefix}rax", np.array([0], dtype=np.int64))]
    )  # (1, k, d_in_i)
    if i == 0:
        x_u = gb.op(
            "Unsqueeze", [x, gb.init(f"{prefix}xax", np.array([1], dtype=np.int64))]
        )  # (B, 1, d_in)
        v = gb.op("Mul", [x_u, r_u])
    else:
        v = gb.op("Mul", [x, r_u])
    q = gb.op("MatMul", [v, wt])  # (B, k, d_block)
    s = gb.init(f"{prefix}s", _f32(arrays[prefix + "s"]))
    b = gb.init(f"{prefix}b", _f32(arrays[prefix + "b"]))
    sax = gb.init(f"{prefix}sax", np.array([0], dtype=np.int64))
    u = gb.op(
        "Add",
        [gb.op("Mul", [q, gb.op("Unsqueeze", [s, sax])]), gb.op("Unsqueeze", [b, sax])],
    )
    return gb.op("Relu", [u])


def _add_tabm_mini_block(gb: _GraphBuilder, cfg, arrays: dict, i: int, x: str) -> str:
    """Shared MLP block; the first block expands members via ``mini_r``."""
    prefix = f"blk{i}_"
    if i == 0:
        mr = gb.init("mini_r", _f32(arrays["mini_r"]))  # (k, d_in)
        mru = gb.op(
            "Unsqueeze", [mr, gb.init("mini_rax", np.array([0], dtype=np.int64))]
        )
        x_u = gb.op(
            "Unsqueeze", [x, gb.init(f"{prefix}xax", np.array([1], dtype=np.int64))]
        )
        x = gb.op("Mul", [x_u, mru])  # (B, k, d_in)
    w = gb.init(f"{prefix}w", _f32(arrays[prefix + "w"]))
    wt = gb.op("Transpose", [w], perm=[1, 0])
    q = gb.op("MatMul", [x, wt])  # (B, k, d_block)
    b = gb.init(f"{prefix}b", _f32(arrays[prefix + "b"]))  # (d_block,)
    return gb.op("Relu", [gb.op("Add", [q, b])])


def _add_tabm_packed_block(gb: _GraphBuilder, cfg, arrays: dict, i: int, x: str) -> str:
    """Fully independent per-member linears."""
    prefix = f"blk{i}_"
    w = gb.init(f"{prefix}w", _f32(arrays[prefix + "w"]))  # (k, d_in_i, d_block)
    b = gb.init(f"{prefix}b", _f32(arrays[prefix + "b"]))  # (k, d_block)
    if i == 0:
        k, d_in = arrays[prefix + "w"].shape[0], arrays[prefix + "w"].shape[1]
        ones = gb.init(f"{prefix}ones", np.ones((1, k, d_in), dtype=np.float32))
        x_u = gb.op(
            "Unsqueeze", [x, gb.init(f"{prefix}xax", np.array([1], dtype=np.int64))]
        )
        x = gb.op("Mul", [x_u, ones])  # (B, k, d_in)
    xt = gb.op("Transpose", [x], perm=[1, 0, 2])  # (k, B, d_in_i)
    qt = gb.op("MatMul", [xt, w])  # (k, B, d_block)
    q = gb.op("Transpose", [qt], perm=[1, 0, 2])  # (B, k, d_block)
    return gb.op("Relu", [gb.op("Add", [q, b])])


_BLOCK_BUILDERS = {
    "tabm": _add_tabm_block,
    "tabm-mini": _add_tabm_mini_block,
    "tabm-packed": _add_tabm_packed_block,
}


def _add_backbone_and_head(gb: _GraphBuilder, cfg, arrays: dict, h: str) -> str:
    """Run the ensemble blocks and average the k member predictions."""
    block_builder = _BLOCK_BUILDERS[cfg.arch_type]
    x = h
    for i in range(cfg.n_blocks):
        x = block_builder(gb, cfg, arrays, i, x)
    xt = gb.op("Transpose", [x], perm=[1, 0, 2])  # (k, B, d_block)
    head_w = gb.init("head_w", _f32(arrays["head_w"]))
    ht = gb.op("MatMul", [xt, head_w])  # (k, B, d_out)
    head_b_u = gb.op(
        "Unsqueeze",
        [
            gb.init("head_b", _f32(arrays["head_b"])),
            gb.init("head_bax", np.array([1], dtype=np.int64)),
        ],
    )  # (k, 1, d_out)
    pt = gb.op("Add", [ht, head_b_u])
    preds = gb.op("Transpose", [pt], perm=[1, 0, 2])  # (B, k, d_out)
    return gb.op(
        "ReduceMean",
        [preds],
        axes=[1],  # axes stays an attribute until ReduceMean-18
        keepdims=0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tabm_to_onnx(
    estimator,
    X=None,
    feature_names=None,
    class_names=None,
    name="ShinrinTabM",
    target_opset=None,
):
    """Export a fitted TabM estimator to an ONNX model proto.

    Parameters
    ----------
    estimator : fitted TabMRegressor or TabMClassifier
        The model to export.
    X : ndarray, optional
        Training-like data used only to sanity-check the number of input
        features; the graph itself accepts any batch size.
    feature_names : list of str, optional
        Stored as model metadata for downstream tooling.
    class_names : list of str, optional
        When provided (classification only), the ``labels`` output yields
        these names instead of integer indices into ``classes_``.
    name : str
        Producer/model name.
    target_opset : int, optional
        Default 15.

    Returns
    -------
    onnx.ModelProto
    """
    if target_opset is None:
        target_opset = DEFAULT_OPSET
    if target_opset < DEFAULT_OPSET:
        raise ValueError(
            f"TabM export requires opset >= {DEFAULT_OPSET}, got {target_opset}"
        )

    if not hasattr(estimator, "params_") or not hasattr(estimator, "preprocessor_"):
        raise ValueError(
            "TabM estimator must be fitted before exporting to ONNX "
            "(missing 'params_'/'preprocessor_' attributes)."
        )

    cfg = estimator.config_
    pre = estimator.preprocessor_
    arrays = estimator.params_.arrays
    n_features_in = int(estimator.n_features_in_)
    if X is not None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != n_features_in:
            raise ValueError(
                f"X has {X.shape[1] if X.ndim == 2 else X.ndim} feature(s); "
                f"expected {n_features_in}"
            )

    gb = _GraphBuilder()

    # -- input ---------------------------------------------------------------
    input_tensor = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [None, n_features_in]
    )

    # -- preprocessing ---------------------------------------------------------
    num_idx = list(pre.numerical_indices_)

    numeric_part = None
    if num_idx:
        x_num_raw = _gather_numeric_columns(gb, "X", num_idx, n_features_in)
        if cfg.use_embeddings:
            # Transforms feed only the piecewise-linear branch; the linear
            # embedding term consumes the raw values (see _Preprocessor).
            x_num_t = _apply_numeric_transforms(gb, pre, x_num_raw, f_idx=0)
            numeric_part = _add_piecewise_linear_embedding(
                gb, cfg, pre, arrays, x_num_raw, x_num_t, f_idx=0
            )
        else:
            # Without embeddings the backbone sees the raw numeric columns
            # (the fitted transforms are only ever consumed by x_enc).
            numeric_part = x_num_raw

    cat_parts = (
        _add_categorical_onehots(gb, pre, "X", tag="cat")
        if pre.categorical_indices_
        else []
    )

    parts: list[str] = []
    if numeric_part is not None:
        parts.append(numeric_part)
    parts.extend(cat_parts)
    if not parts:  # pragma: no cover - n_features >= 1 guarantees a part
        raise ValueError("TabM model has no input features")
    if len(parts) == 1:
        h = parts[0]
    else:
        h = gb.op("Concat", parts, axis=1)

    # -- backbone + averaged head ---------------------------------------------
    avg = _add_backbone_and_head(gb, cfg, arrays, h)

    # -- task-specific tails ----------------------------------------------------
    is_classifier = getattr(estimator, "out_activation_", "identity") != "identity"
    outputs: list = []

    if not is_classifier:
        if cfg.d_out == 1:
            # Match the estimator's sklearn-compatible single-output shape.
            gb.op(
                "Squeeze",
                [avg, gb.init("squeeze_ax", np.array([1], dtype=np.int64))],
                output="predictions",
            )
        else:
            gb.op("Identity", [avg], output="predictions")
        out_dims = [None] if cfg.d_out == 1 else [None, cfg.d_out]
        outputs.append(
            helper.make_tensor_value_info("predictions", TensorProto.FLOAT, out_dims)
        )
    else:
        if estimator.out_activation_ == "logistic":
            col = gb.op(
                "Slice",
                [
                    avg,
                    gb.init("bin_start", np.array([0], dtype=np.int64)),
                    gb.init("bin_end", np.array([1], dtype=np.int64)),
                    gb.init("bin_ax", np.array([1], dtype=np.int64)),
                ],
            )
            p = gb.op("Sigmoid", [col])
            one = gb.konst("proba_one", np.array(1.0, dtype=np.float32))
            inv = gb.op("Sub", [one, p])
            gb.op("Concat", [inv, p], output="probabilities", axis=1)
        else:
            gb.op("Softmax", [avg], output="probabilities", axis=1)
        n_cols = 2 if estimator.out_activation_ == "logistic" else cfg.d_out
        outputs.append(
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, n_cols]
            )
        )
        argmax = gb.op("ArgMax", ["probabilities"], axis=1, keepdims=0)
        if class_names is not None:
            names = np.asarray([str(c) for c in class_names], dtype=np.str_)
            names_i = gb.init("class_names", names)
            gb.op("Gather", [names_i, argmax], output="labels", axis=0)
            outputs.append(
                helper.make_tensor_value_info("labels", TensorProto.STRING, [None])
            )
        else:
            gb.op("Identity", [argmax], output="labels")
            outputs.append(
                helper.make_tensor_value_info("labels", TensorProto.INT64, [None])
            )

    graph = helper.make_graph(
        gb.nodes,
        f"{name}_graph",
        [input_tensor],
        outputs,
        initializer=gb.initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", target_opset)],
        producer_name=name,
        ir_version=8,
    )
    props: dict[str, str] = {
        "shinrin_version": "0.2.0",
        "model_type": "TabMClassifier" if is_classifier else "TabMRegressor",
        "arch_type": cfg.arch_type,
        "k": str(cfg.k),
    }
    if feature_names is not None:
        props["feature_names"] = ",".join(str(f) for f in feature_names)
    helper.set_model_props(model, props)
    return model
