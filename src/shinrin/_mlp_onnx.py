"""ONNX exporter for shinrin MLPClassifier / MLPRegressor.

Converts a fitted :class:`~shinrin.mlp.MLPClassifier` /
:class:`~shinrin.mlp.MLPRegressor` into a self-contained ONNX graph.
The exported graph reproduces the full inference pipeline — preprocessing
(quantile / asinh / standard-scaler transforms, piecewise-linear encoding,
categorical one-hot), the optional PLE embedding layer, the MLP backbone
and the output activation — so deployment needs nothing but the ``.onnx``
file and raw feature vectors.

Design notes
------------
- Only standard-domain ops are emitted (default opset 15, matching
  :mod:`shinrin.onnx`); the ``ai.onnx.ml`` domain is not used.
- Quantization (ternary / BitLinear) is baked into the exported weights
  so the ONNX graph always runs at full float32 precision.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

__all__ = ["mlp_to_onnx"]

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
# Preprocessing subgraphs  (mirrors _Preprocessor.transform + fit)
# ---------------------------------------------------------------------------


def _gather_numeric_columns(
    gb: _GraphBuilder,
    x: str,
    num_indices: list[int],
    n_features_in: int,
) -> str | None:
    if not num_indices:
        return None
    if len(num_indices) == n_features_in and num_indices == list(range(n_features_in)):
        return x
    idx = gb.init("num_col_idx", np.array(num_indices, dtype=np.int64))
    return gb.op("Gather", [x, idx], axis=1)


def _apply_numeric_transforms(
    gb: _GraphBuilder,
    pre,
    x: str,
    f_idx: str,
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
            raise ValueError(f"Unsupported MLP transform: {kind}")
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

    Mirrors ``_Preprocessor.transform`` + ``MLPCore._embed_forward``:
    the *linear* term consumes the raw numerical features while the
    piecewise-linear branch consumes the *transformed* features, matching
    the training-time layout.
    """
    bins = pre.bins_
    n_feat = len(bins)
    demb = cfg.d_embedding
    max_bins = max(len(edges) - 1 for edges in bins)

    # Padded per-feature tables.
    edges_lo = np.zeros((n_feat, max_bins), dtype=np.float32)
    widths = np.ones((n_feat, max_bins), dtype=np.float32)
    rules = np.zeros((n_feat, max_bins), dtype=np.int64)
    wp_all = np.zeros((n_feat, max_bins, demb), dtype=np.float32)
    for f, edges in enumerate(bins):
        m = len(edges) - 1
        diffs = np.diff(edges).astype(np.float32)
        edges_lo[f, :m] = edges[:-1]
        widths[f, :m] = diffs
        if m == 1:
            rules[f, 0] = 4
        else:
            rules[f, 0] = 1
            rules[f, 1 : m - 1] = 2
            rules[f, m - 1] = 3
        wp_all[f, :m] = params_arrays[f"emb_wp_{f}"]

    edges_i = gb.init(f"ple_edges_lo_{f_idx}", edges_lo)
    widths_i = gb.init(f"ple_widths_{f_idx}", widths)
    rules_i = gb.init(f"ple_rules_{f_idx}", rules)

    x3 = gb.op(
        "Unsqueeze",
        [x_num_t, gb.init(f"ple_ax2_{f_idx}", np.array([2], dtype=np.int64))],
    )
    t = gb.op("Div", [gb.op("Sub", [x3, edges_i]), widths_i])

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
    enc = gb.op("Mul", [sel3, valid])

    enc_t = gb.op("Transpose", [enc], perm=[1, 0, 2])
    pl_t = gb.op("MatMul", [enc_t, gb.init(f"ple_wp_{f_idx}", wp_all)])
    pl = gb.op("Transpose", [pl_t], perm=[1, 0, 2])

    w0 = gb.init(f"emb_w0_{f_idx}", _f32(params_arrays["emb_w0"]))
    b0 = gb.init(f"emb_b0_{f_idx}", _f32(params_arrays["emb_b0"]))
    raw3 = gb.op(
        "Unsqueeze",
        [x_num_raw, gb.init(f"raw_ax2_{f_idx}", np.array([2], dtype=np.int64))],
    )
    lin0 = gb.op("Add", [gb.op("Mul", [raw3, w0]), b0])
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
        )
        vals = gb.init(f"{tag}_vals_{pos}", values.reshape(1, card))
        le = gb.op("LessOrEqual", [vals, col])
        counts = gb.op(
            "ReduceSum",
            [
                gb.op("Cast", [le], to=TensorProto.FLOAT),
                gb.init(f"{tag}_rax_{pos}", np.array([1], dtype=np.int64)),
            ],
            keepdims=0,
        )
        one_f = gb.konst(f"{tag}_one_{pos}", np.array(1.0, dtype=np.float32))
        zero = gb.konst(f"{tag}_zero_{pos}", np.array(0.0, dtype=np.float32))
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
# Main exporter
# ---------------------------------------------------------------------------


def mlp_to_onnx(
    estimator,
    X=None,
    feature_names=None,
    class_names=None,
    name="ShinrinMLP",
    target_opset=DEFAULT_OPSET,
):
    """Export a fitted MLPClassifier or MLPRegressor to ONNX.

    Returns
    -------
    onnx.ModelProto
    """
    n_features_in = int(getattr(estimator, "n_features_in_", 0))
    if n_features_in == 0 and X is not None:
        n_features_in = X.shape[1]

    pre = estimator.preprocessor_
    cfg = estimator.config_
    arrays = estimator.params_.arrays
    out_activation = estimator.out_activation_
    n_outputs = int(estimator.n_outputs_)
    is_classifier = hasattr(estimator, "classes_")

    gb = _GraphBuilder()

    # --- Input ---------------------------------------------------------------
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features_in])
    ]

    # --- Numeric column gathering --------------------------------------------
    x_num_raw = _gather_numeric_columns(gb, "X", pre.numerical_indices_, n_features_in)

    # --- Numeric transforms --------------------------------------------------
    if x_num_raw is not None:
        x_num_t = x_num_raw
        for ti, _ in enumerate(pre.transforms_):
            x_num_t = _apply_numeric_transforms(gb, pre, x_num_t, f"t{ti}")
    else:
        x_num_t = None

    # --- PLE embedding (if enabled) ------------------------------------------
    emb_flat = None
    if cfg.use_embeddings and cfg.n_num_features and x_num_raw is not None:
        assert x_num_t is not None
        emb_flat = _add_piecewise_linear_embedding(
            gb, cfg, pre, arrays, x_num_raw, x_num_t, 0
        )

    # --- Categorical one-hot encoding ----------------------------------------
    cat_parts = []
    if pre.categorical_indices_:
        cat_parts = _add_categorical_onehots(gb, pre, "X", "cat")

    # --- Concatenate to form MLP input ---------------------------------------
    parts: list[str] = []
    if emb_flat is not None:
        parts.append(emb_flat)
    elif x_num_raw is not None:
        parts.append(x_num_raw)
    parts.extend(cat_parts)

    if not parts:
        raise ValueError("MLP has no numerical or categorical features to build input.")
    if len(parts) == 1:
        h = parts[0]
    else:
        h = gb.op("Concat", parts, axis=1)

    # --- MLP forward pass ----------------------------------------------------
    # Apply ternary quantization to weights at export time.
    for i in range(cfg.n_layers):
        w_name = f"l{i}_w"
        w_raw = _f32(arrays[w_name])
        if cfg.quantization != "none" and cfg.layer_is_quantized(i):
            from shinrin._quant import ternary_quantize_dequantize

            w_raw = ternary_quantize_dequantize(w_raw, cfg.quantization_granularity)
        w_i = gb.init(w_name, w_raw)
        # Weights stored (d_out, d_in); transpose for ONNX MatMul (x @ W.T)
        wt = gb.op("Transpose", [w_i], perm=[1, 0])
        b_i = gb.init(f"l{i}_b", _f32(arrays[f"l{i}_b"]))
        z = gb.op("Add", [gb.op("MatMul", [h, wt]), b_i])
        if i < cfg.n_layers - 1:  # hidden layer
            act_map = {
                "relu": "Relu",
                "tanh": "Tanh",
                "logistic": "Sigmoid",
                "identity": None,
            }
            op = act_map.get(cfg.activation)
            h = gb.op(op, [z]) if op else z
        else:
            h = z  # output layer: linear

    # --- Output post-processing ----------------------------------------------
    if not is_classifier:
        # Regression: squeeze to (N,) for single-output or keep (N, n_outputs)
        if n_outputs == 1:
            gb.op(
                "Squeeze",
                [h, gb.init("out_ax", np.array([1], dtype=np.int64))],
                output="predictions",
            )
            outputs = [
                helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None])
            ]
        else:
            gb.op("Identity", [h], output="predictions")
            outputs = [
                helper.make_tensor_value_info(
                    "predictions", TensorProto.FLOAT, [None, n_outputs]
                )
            ]
    elif out_activation == "logistic":
        # Binary: sigmoid → [1-p, p]
        p = gb.op("Sigmoid", [h])
        one = gb.konst("one", np.array(1.0, dtype=np.float32))
        one_minus_p = gb.op("Sub", [one, p])
        proba = gb.op("Concat", [one_minus_p, p], output="probabilities", axis=1)
        # Derive labels from the probabilities themselves (sklearn's
        # predict is argmax(predict_proba)); a threshold on the raw
        # sigmoid column would emit (n, 1) labels that can disagree with
        # the reported probabilities.
        gb.op("ArgMax", [proba], output="labels", axis=1, keepdims=0)
        outputs = [
            helper.make_tensor_value_info("labels", TensorProto.INT64, [None]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, 2]
            ),
        ]
    else:
        # Multiclass: softmax
        # ReduceMax(opset 15): axes is an attribute; ReduceSum: axes is an input
        red_ax1_init = gb.init("red_ax1", np.array([1], dtype=np.int64))
        shifted = gb.op("Sub", [h, gb.op("ReduceMax", [h], keepdims=1, axes=[1])])
        exp = gb.op("Exp", [shifted])
        sum_exp = gb.op("ReduceSum", [exp, red_ax1_init], keepdims=1)
        proba = gb.op("Div", [exp, sum_exp], output="probabilities")
        gb.op("ArgMax", [proba], output="labels", axis=1, keepdims=0)
        outputs = [
            helper.make_tensor_value_info("labels", TensorProto.INT64, [None]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, n_outputs]
            ),
        ]

    # --- Build model ---------------------------------------------------------
    opset_imports = [
        helper.make_opsetid("", target_opset),
    ]
    graph = helper.make_graph(
        gb.nodes,
        f"{name}_graph",
        inputs,
        outputs,
        initializer=gb.initializers,
    )
    model = helper.make_model(
        graph, opset_imports=opset_imports, producer_name=name, ir_version=8
    )
    return model
