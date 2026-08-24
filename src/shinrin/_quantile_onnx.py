"""Exact ONNX export for quantile regression trees and forests.

ONNX graphs are static, so the quantile must be chosen at export time and
is baked into the graph (pass ``quantile=`` to :func:`shinrin.onnx.to_onnx`).

Single trees
------------
``BaseTreeQuantileRegressor.predict(X, quantile)`` routes each sample to a
leaf and returns the weighted percentile of the training ``y`` values that
landed in that leaf.  The per-leaf percentile is a constant once the
quantile is fixed, so the export simply precomputes it for every leaf and
emits a plain ``ai.onnx.ml`` ``TreeEnsembleRegressor`` whose leaf outputs
are those constants.

Forests
-------
``BaseForestQuantileRegressor.predict(X, quantile)`` pools the bootstrap
weights of every training sample that shares a leaf with ``x`` in *any*
tree and takes a weighted percentile over the full training target vector.
Because the pooled weights depend on which combination of leaves ``x``
reaches, they are computed inside the graph:

1. one ``TreeEnsembleRegressor`` per tree emits a compact leaf index,
2. a per-tree ``(n_leaves, n_train)`` matrix scatters the tree's
   normalised bootstrap counts (``y_weights_``) onto training samples,
3. the summed weight vector is reordered by the static ``argsort``
   permutation of ``y_train_`` via a fixed permutation matrix, and
4. a small sub-graph reproduces ``weighted_percentile`` exactly, including
   its removal of zero-weight samples and its midpoint interpolation
   between neighbouring order statistics.

The percentile tail runs in float32 while the reference implementation
performs its final interpolation in float64, so forest results match native
``predict`` within float32 rounding (~1e-6 relative) rather than bitwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover
    TensorProto: Any = None
    helper: Any = None

# ai.onnx.ml domain version used for the tree ensembles (see onnx.py).
AI_ONNX_ML_OPSET = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_onnx() -> None:
    if helper is None or TensorProto is None:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )


def _validate_quantile(quantile) -> float:
    """Return the quantile as a float, rejecting ``None``."""
    if quantile is None:
        raise ValueError(
            "Quantile models must be exported with an explicit `quantile` "
            "(0-100) baked into the graph. Runtime-selectable quantiles "
            "cannot be represented in a static ONNX graph."
        )
    quantile = float(quantile)
    if quantile < 0 or quantile > 100:
        raise ValueError(f"q should be in-between 0 and 100, got {quantile}")
    return quantile


def _tree_ensemble_attrs(tree, leaf_values: dict[int, float]) -> dict:
    """Build ``TreeEnsembleRegressor`` attributes emitting one value per leaf.

    ``leaf_values`` maps leaf node id -> emitted float. Internal nodes use
    ``BRANCH_LEQ`` (x <= threshold goes to the true/left child), matching
    sklearn's routing, so ensemble outputs agree with ``tree_.apply``.
    """
    left = np.asarray(tree.children_left)
    right = np.asarray(tree.children_right)
    feat = np.asarray(tree.feature)
    thr = np.asarray(tree.threshold, dtype=np.float64)
    n_nodes = len(left)
    internal = left >= 0

    leaf_ids = np.array(sorted(leaf_values), dtype=np.int64)
    n_leaves = len(leaf_ids)

    return {
        "nodes_treeids": np.zeros(n_nodes, dtype=np.int64),
        "nodes_nodeids": np.arange(n_nodes, dtype=np.int64),
        "nodes_featureids": np.where(internal, feat, 0).astype(np.int64),
        "nodes_modes": ["BRANCH_LEQ" if b else "LEAF" for b in internal],
        "nodes_values": np.where(internal, thr, 0.0).astype(np.float32),
        "nodes_truenodeids": np.where(internal, left, 0).astype(np.int64),
        "nodes_falsenodeids": np.where(internal, right, 0).astype(np.int64),
        "nodes_missing_value_tracks_true": np.zeros(n_nodes, dtype=np.int64),
        "n_targets": 1,
        "aggregate_function": "SUM",
        "post_transform": "NONE",
        "target_treeids": np.zeros(n_leaves, dtype=np.int64),
        "target_nodeids": leaf_ids,
        "target_ids": np.zeros(n_leaves, dtype=np.int64),
        "target_weights": np.array(
            [leaf_values[int(nid)] for nid in leaf_ids], dtype=np.float32
        ),
    }


def _make_model(nodes, inits, inputs, outputs, name: str, target_opset: int):
    graph = helper.make_graph(
        nodes, f"{name}_graph", inputs, outputs, initializer=inits
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", target_opset),
            helper.make_opsetid("ai.onnx.ml", AI_ONNX_ML_OPSET),
        ],
        producer_name=name,
        ir_version=8,
    )


def _add_init(inits, nm, arr) -> None:
    arr = np.ascontiguousarray(arr)
    inits.append(
        helper.make_tensor(
            nm, TensorProto.FLOAT, list(arr.shape), arr.astype(np.float32).ravel()
        )
    )


def _add_init_i64(inits, nm, arr) -> None:
    arr = np.ascontiguousarray(arr)
    inits.append(
        helper.make_tensor(
            nm, TensorProto.INT64, list(arr.shape), arr.astype(np.int64).ravel()
        )
    )


# ---------------------------------------------------------------------------
# Single trees
# ---------------------------------------------------------------------------


def quantile_tree_to_onnx(
    estimator,
    quantile=None,
    feature_names=None,
    name="ShinrinQuantileTree",
    target_opset=15,
):
    """Export a fitted quantile tree with ``quantile`` baked in.

    Regression output: ``predictions`` of shape ``(batch,)``, matching
    ``estimator.predict(X, quantile)``.
    """
    _require_onnx()
    quantile = _validate_quantile(quantile)

    from shinrin._skgarden.quantile.utils import weighted_percentile

    y = np.asarray(estimator.y_train_)
    y_leaves = np.asarray(estimator.y_train_leaves_)
    leaf_values = {
        int(leaf): float(weighted_percentile(y[y_leaves == leaf], quantile))
        for leaf in np.unique(y_leaves)
    }
    attrs = _tree_ensemble_attrs(estimator.tree_, leaf_values)

    n_features = int(estimator.tree_.n_features)
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    inits: list = []
    nodes = [
        helper.make_node(
            "TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["te_out"],
            domain="ai.onnx.ml",
            **attrs,
        )
    ]
    # Flatten ORT's (N, 1) tree-ensemble output to sklearn's (N,).
    _add_init_i64(inits, "neg1", [-1])
    nodes.append(helper.make_node("Reshape", ["te_out", "neg1"], ["predictions"]))
    outputs = [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None])]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)


# ---------------------------------------------------------------------------
# Forests
# ---------------------------------------------------------------------------


def _forest_weight_matrices(estimator) -> list[tuple[np.ndarray, dict[int, int]]]:
    """Per-tree bootstrap-weight matrices plus their compact leaf mappings.

    Row ``k`` of each ``(n_leaves_i, n_train)`` matrix holds
    ``y_weights_[i, j]`` for every training sample ``j`` whose leaf in tree
    ``i`` is the k-th distinct (non-sentinel) leaf. Samples never drawn by
    tree ``i``'s bootstrap keep weight zero in every row.
    """
    y_leaves = np.asarray(estimator.y_train_leaves_)
    y_weights = np.asarray(estimator.y_weights_, dtype=np.float32)
    n_train = y_weights.shape[1]

    mats: list[tuple[np.ndarray, dict[int, int]]] = []
    for i in range(y_leaves.shape[0]):
        row = y_leaves[i]
        uniq = np.unique(row[row >= 0])
        compact = {int(leaf): k for k, leaf in enumerate(uniq)}
        mat = np.zeros((len(uniq), n_train), dtype=np.float32)
        mask = row >= 0
        rows = np.array([compact[int(leaf)] for leaf in row[mask]], dtype=np.int64)
        mat[rows, np.flatnonzero(mask)] = y_weights[i][mask]
        mats.append((mat, compact))
    return mats


def _percentile_tail(
    nodes: list,
    inits: list,
    weights_name: str,
    y_sorted: np.ndarray,
    quantile: float,
    suffix: str = "",
) -> str:
    """Emit a sub-graph computing the weighted percentile per sample row.

    ``weights_name`` is a ``(batch, n_train)`` float32 tensor of pooled
    sample weights; ``y_sorted`` is ``y_train[argsort(y_train)]`` so that
    column order matches the reference's sorted view. Returns the name of
    the resulting ``(batch,)`` tensor.

    The reference implementation removes zero-weight samples before
    selecting interpolation knots; this tail reproduces that semantics by
    requiring knots to carry positive weight, so knot choices agree exactly
    and results differ only through float32 arithmetic.
    """
    n = y_sorted.size
    s = suffix
    ar_f = np.arange(n, dtype=np.float32)
    _add_init(inits, f"{s}ysorted", y_sorted.reshape(-1))
    _add_init(inits, f"{s}ar", ar_f)
    _add_init(inits, f"{s}ones_col", np.ones((n, 1), dtype=np.float32))
    _add_init(inits, f"{s}q", [float(quantile)])
    _add_init(inits, f"{s}zero", [0.0])
    _add_init(inits, f"{s}half", [0.5])
    _add_init(inits, f"{s}one", [1.0])
    _add_init(inits, f"{s}hundred", [100.0])
    _add_init(inits, f"{s}big", [np.float32(3.0e9)])
    _add_init(inits, f"{s}nmax", [np.float32(n - 1)])
    _add_init_i64(inits, f"{s}ax1", [1])
    _add_init_i64(inits, f"{s}axis", [1])

    # Total pooled weight per row: (batch, 1).
    nodes.append(
        helper.make_node("MatMul", [weights_name, f"{s}ones_col"], [f"{s}total"])
    )
    # partial_sum = 100 / total * (cumsum(w) - w / 2), matching the
    # reference's midpoint placement of each sample inside its weight.
    nodes.append(helper.make_node("CumSum", [weights_name, f"{s}axis"], [f"{s}cs"]))
    nodes.append(helper.make_node("Div", [f"{s}hundred", f"{s}total"], [f"{s}scale"]))
    nodes.append(helper.make_node("Mul", [weights_name, f"{s}half"], [f"{s}wh"]))
    nodes.append(helper.make_node("Sub", [f"{s}cs", f"{s}wh"], [f"{s}csm"]))
    nodes.append(helper.make_node("Mul", [f"{s}csm", f"{s}scale"], [f"{s}pw"]))

    # valid = weight > 0; candidate lower knot A = valid & (pw < q).
    nodes.append(helper.make_node("Greater", [weights_name, f"{s}zero"], [f"{s}valid"]))
    nodes.append(helper.make_node("Less", [f"{s}pw", f"{s}q"], [f"{s}ltq"]))
    nodes.append(helper.make_node("And", [f"{s}valid", f"{s}ltq"], [f"{s}cand"]))
    nodes.append(
        helper.make_node("Cast", [f"{s}valid"], [f"{s}vf"], to=TensorProto.FLOAT)
    )
    nodes.append(
        helper.make_node("Cast", [f"{s}cand"], [f"{s}cf"], to=TensorProto.FLOAT)
    )

    # Per-row counts: number of candidates and number of valid samples.
    nodes.append(helper.make_node("MatMul", [f"{s}cf", f"{s}ones_col"], [f"{s}cnta"]))
    nodes.append(helper.make_node("MatMul", [f"{s}vf", f"{s}ones_col"], [f"{s}tv"]))

    # Branch flags: has_lo -> a lower knot exists; has_hi -> a higher knot
    # exists after it (reference's start == -1 / start == last branches).
    nodes.append(helper.make_node("Greater", [f"{s}cnta", f"{s}zero"], [f"{s}has_lo"]))
    nodes.append(helper.make_node("Less", [f"{s}cnta", f"{s}tv"], [f"{s}has_hi"]))
    nodes.append(
        helper.make_node("And", [f"{s}has_lo", f"{s}has_hi"], [f"{s}interp_ok"])
    )

    # Lower knot position: largest candidate index (0 when none; unused then).
    nodes.append(helper.make_node("Mul", [f"{s}cf", f"{s}ar"], [f"{s}cpos"]))
    nodes.append(
        helper.make_node("ReduceMax", [f"{s}cpos"], [f"{s}start"], axes=[1], keepdims=1)
    )

    # Upper knot position: first index whose inclusive count of VALID
    # samples equals cnta + 1. When cnta == 0 this selects the first valid
    # sample, which reproduces the reference's `return sorted_a[0]` branch.
    nodes.append(helper.make_node("CumSum", [f"{s}vf", f"{s}axis"], [f"{s}csv"]))
    nodes.append(helper.make_node("Add", [f"{s}cnta", f"{s}one"], [f"{s}want"]))
    nodes.append(helper.make_node("Equal", [f"{s}csv", f"{s}want"], [f"{s}eqnext"]))
    nodes.append(
        helper.make_node("Cast", [f"{s}eqnext"], [f"{s}eqf"], to=TensorProto.FLOAT)
    )
    nodes.append(helper.make_node("Sub", [f"{s}one", f"{s}eqf"], [f"{s}noteq"]))
    nodes.append(helper.make_node("Mul", [f"{s}noteq", f"{s}big"], [f"{s}bign"]))
    nodes.append(helper.make_node("Add", [f"{s}bign", f"{s}ar"], [f"{s}npos_full"]))
    nodes.append(
        helper.make_node(
            "ReduceMin", [f"{s}npos_full"], [f"{s}next_raw"], axes=[1], keepdims=1
        )
    )
    # Clamp so the Gather stays in bounds even when no upper knot exists;
    # the gathered value is discarded by the Where below in that case.
    nodes.append(helper.make_node("Min", [f"{s}next_raw", f"{s}nmax"], [f"{s}next"]))

    # Last valid position (upper boundary branch value).
    nodes.append(helper.make_node("Mul", [f"{s}vf", f"{s}ar"], [f"{s}lvpos"]))
    nodes.append(
        helper.make_node(
            "ReduceMax", [f"{s}lvpos"], [f"{s}lastv"], axes=[1], keepdims=1
        )
    )

    # Integer index tensors for gathers: (batch, 1) each.
    nodes.append(
        helper.make_node("Cast", [f"{s}start"], [f"{s}start_i"], to=TensorProto.INT64)
    )
    nodes.append(
        helper.make_node("Cast", [f"{s}next"], [f"{s}next_i"], to=TensorProto.INT64)
    )
    nodes.append(
        helper.make_node("Cast", [f"{s}lastv"], [f"{s}lastv_i"], to=TensorProto.INT64)
    )

    # Sorted y values at both knots.
    nodes.append(
        helper.make_node("Gather", [f"{s}ysorted", f"{s}start_i"], [f"{s}lo_v"], axis=0)
    )
    nodes.append(
        helper.make_node("Gather", [f"{s}ysorted", f"{s}next_i"], [f"{s}hi_v"], axis=0)
    )
    # Partial sums at both knots (per-row gather along axis 1).
    nodes.append(
        helper.make_node(
            "GatherElements", [f"{s}pw", f"{s}start_i"], [f"{s}lo_p"], axis=1
        )
    )
    nodes.append(
        helper.make_node(
            "GatherElements", [f"{s}pw", f"{s}next_i"], [f"{s}hi_p"], axis=1
        )
    )

    # Midpoint interpolation between neighbouring order statistics.
    nodes.append(helper.make_node("Sub", [f"{s}q", f"{s}lo_p"], [f"{s}dnum"]))
    nodes.append(helper.make_node("Sub", [f"{s}hi_p", f"{s}lo_p"], [f"{s}dden"]))
    nodes.append(helper.make_node("Div", [f"{s}dnum", f"{s}dden"], [f"{s}frac"]))
    nodes.append(helper.make_node("Sub", [f"{s}hi_v", f"{s}lo_v"], [f"{s}dv"]))
    nodes.append(helper.make_node("Mul", [f"{s}frac", f"{s}dv"], [f"{s}dfac"]))
    nodes.append(helper.make_node("Add", [f"{s}lo_v", f"{s}dfac"], [f"{s}interp_val"]))

    # Upper boundary value: last positive-weight sample.
    nodes.append(
        helper.make_node(
            "Gather", [f"{s}ysorted", f"{s}lastv_i"], [f"{s}last_val"], axis=0
        )
    )

    # When has_lo is false the upper knot is the first valid sample, and
    # hi_v holds exactly that value (reference's sorted_a[0]).
    nodes.append(
        helper.make_node(
            "Where", [f"{s}has_lo", f"{s}last_val", f"{s}hi_v"], [f"{s}edge_val"]
        )
    )
    nodes.append(
        helper.make_node(
            "Where", [f"{s}interp_ok", f"{s}interp_val", f"{s}edge_val"], [f"{s}res2d"]
        )
    )
    nodes.append(helper.make_node("Squeeze", [f"{s}res2d", f"{s}ax1"], [f"{s}res"]))
    return f"{s}res"


def quantile_forest_to_onnx(
    estimator,
    quantile=None,
    feature_names=None,
    name="ShinrinQuantileForest",
    target_opset=15,
):
    """Export a fitted quantile forest with ``quantile`` baked in.

    Regression output: ``predictions`` of shape ``(batch,)``, matching
    ``estimator.predict(X, quantile)``.
    """
    _require_onnx()
    quantile = _validate_quantile(quantile)

    y = np.asarray(estimator.y_train_)
    n_train = y.size
    sorter = np.argsort(y)
    y_sorted = np.asarray(y, dtype=np.float32)[sorter]

    n_features = int(estimator.estimators_[0].tree_.n_features)
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    inits: list = []
    nodes: list = []

    w_terms: list[str] = []
    for i, (mat, compact) in enumerate(_forest_weight_matrices(estimator)):
        attrs = _tree_ensemble_attrs(
            estimator.estimators_[i].tree_,
            {leaf: float(k) for leaf, k in compact.items()},
        )
        out = f"leaf{i}"
        nodes.append(
            helper.make_node(
                "TreeEnsembleRegressor",
                inputs=["X"],
                outputs=[out],
                domain="ai.onnx.ml",
                **attrs,
            )
        )
        _add_init(inits, f"lv{i}", np.arange(len(compact), dtype=np.float32))
        nodes.append(helper.make_node("Equal", [out, f"lv{i}"], [f"ohb{i}"]))
        nodes.append(
            helper.make_node("Cast", [f"ohb{i}"], [f"ohf{i}"], to=TensorProto.FLOAT)
        )
        _add_init(inits, f"m{i}", mat)
        wname = f"w{i}"
        nodes.append(helper.make_node("MatMul", [f"ohf{i}", f"m{i}"], [wname]))
        w_terms.append(wname)

    if len(w_terms) > 1:
        acc = w_terms[0]
        for k, term in enumerate(w_terms[1:]):
            out = f"wacc{k}"
            nodes.append(helper.make_node("Add", [acc, term], [out]))
            acc = out
        w_name = acc
    else:
        w_name = w_terms[0]

    # Reorder weight columns into argsort(y) order with a permutation matrix.
    perm = np.zeros((n_train, n_train), dtype=np.float32)
    perm[sorter, np.arange(n_train)] = 1.0
    _add_init(inits, "perm", perm)
    nodes.append(helper.make_node("MatMul", [w_name, "perm"], ["wsorted"]))

    res = _percentile_tail(nodes, inits, "wsorted", y_sorted, quantile)
    nodes.append(helper.make_node("Identity", [res], ["predictions"]))
    outputs = [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None])]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)
