"""ONNX exporter for shinrin tree, forest, and TabM models.

This module provides functionality to export fitted shinrin tree, forest,
and TabM models to the ONNX format, enabling deployment on platforms that
support ONNX runtime inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy import for ONNX types (used in _onnx_dtype)
try:
    from onnx import TensorProto
except ImportError:  # pragma: no cover
    TensorProto: Any = None


# ai.onnx.ml opset that introduced the standalone ``TreeEnsemble`` operator
# used for all tree/forest exports. Runtimes must support this opset (or
# newer) to load the exported models.
TREE_ENSEMBLE_OPSET = 5

# Model metadata property marking exports emitted by this module. The value
# records the leaf-value encoding so :mod:`shinrin.onnx_import` can parse
# models without guessing.
_ENCODING_PROP = "shinrin_tree_encoding"
_TASK_PROP = "shinrin_task"
_BASE_VALUES_PROP = "shinrin_base_values"

# Leaf-value encodings recorded in model metadata:
#   "probs-v5"            classification, per-class probability rows
#   "logit-v5"            classification, per-class log-odds columns
#   "mean-v5"             regression, per-leaf means
_ENCODING_PROBS = "probs-v5"
_ENCODING_LOGIT = "logit-v5"
_ENCODING_MEAN = "mean-v5"


def _tree_info(tree) -> dict[str, Any]:
    """Extract a sklearn-style ``tree_`` into plain arrays.

    Returns a dict with:

    - ``feature``: split feature ids, ``-2`` marking leaf nodes
    - ``threshold``: split thresholds (``0`` at leaves)
    - ``left``/``right``: child node ids (leaves point at ``-1``)
    - ``value``: leaf prediction targets — regression means shaped
      ``(n_nodes,)`` or class counts shaped ``(n_nodes, n_classes)``;
      non-leaf rows are zeroed
    - ``n_classes``: number of classes (1 for regression)
    """
    t = tree.tree_
    feature = t.feature.astype(np.int64)
    leaf = feature == -2
    left = t.children_left.astype(np.int64)
    right = t.children_right.astype(np.int64)
    n_classes = int(t.n_classes[0])
    if n_classes == 1:
        value = np.where(leaf, t.value.ravel(), 0.0).astype(np.float64)
    else:
        counts = t.value[:, 0, :n_classes].astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        value = np.where(leaf.reshape(-1, 1), counts / totals, 0.0)
    return {
        "feature": feature,
        "threshold": t.threshold.astype(np.float64),
        "left": left,
        "right": right,
        "value": value,
        "n_classes": n_classes,
    }


def _ensemble_attributes(
    entries: list[tuple[dict[str, Any], int]],
    n_targets: int,
    dtype: np.dtype,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Build attributes for one ``ai.onnx.ml.TreeEnsemble`` node (opset 5).

    ``entries`` pairs tree-info dicts with a fixed target column whose leaf
    weights contribute to. The v5 operator addresses leaves through compact
    parallel ``leaf_weights``/``leaf_targetids`` arrays where every entry is
    one ``(weight, target)`` tuple; branch ids index the node arrays unless
    the matching ``nodes_trueleafs``/``nodes_falseleafs`` flag selects the
    leaf-entry array instead. Aggregation is always ``SUM``; callers fold
    averaging factors (``1 / n_trees``) or boosting learning rates into the
    leaf weights beforehand.
    """
    from onnx import numpy_helper

    dt = np.dtype(dtype)
    feats: list[int] = []
    splits: list[float] = []
    true_ids: list[int] = []
    true_leafs: list[int] = []
    false_ids: list[int] = []
    false_leafs: list[int] = []
    leaf_w: list[float] = []
    leaf_t: list[int] = []
    roots: list[int] = []
    node_off = 0
    entry_off = 0

    for info, target in entries:
        feature = info["feature"]
        leaf = feature == -2
        interior_idx = np.flatnonzero(~leaf)
        leaf_idx = np.flatnonzero(leaf)
        rank = {int(orig): p for p, orig in enumerate(interior_idx)}
        entry_of_leaf = {int(orig): k for k, orig in enumerate(leaf_idx)}
        values = info["value"]

        n_interior = len(interior_idx)
        if n_interior == 0:
            # Degenerate tree without splits: synthesize a root whose two
            # branches both resolve to the single leaf entry.
            feats.append(0)
            splits.append(0.0)
            true_ids.append(entry_off)
            true_leafs.append(1)
            false_ids.append(entry_off)
            false_leafs.append(1)
            leaf_w.append(float(values.ravel()[leaf_idx[0]]) * scale)
            leaf_t.append(target)
            roots.append(node_off)
            node_off += 1
            entry_off += 1
            continue

        for orig in interior_idx:
            l_orig = int(info["left"][orig])
            r_orig = int(info["right"][orig])
            tl = bool(leaf[l_orig])
            fl = bool(leaf[r_orig])
            feats.append(int(feature[orig]))
            splits.append(float(info["threshold"][orig]))
            true_ids.append(entry_of_leaf[l_orig] + entry_off if tl else rank[l_orig] + node_off)
            true_leafs.append(int(tl))
            false_ids.append(entry_of_leaf[r_orig] + entry_off if fl else rank[r_orig] + node_off)
            false_leafs.append(int(fl))

        leaf_vals = values[leaf_idx]
        for k in range(len(leaf_idx)):
            w = float(leaf_vals[k, target] if leaf_vals.ndim == 2 else leaf_vals[k])
            leaf_w.append(w * scale)
            leaf_t.append(target)

        roots.append(node_off)
        node_off += n_interior
        entry_off += len(leaf_idx)

    return {
        "name": "shinrin_tree_ensemble",
        "domain": "ai.onnx.ml",
        "tree_roots": np.array(roots, dtype=np.int64),
        "nodes_featureids": np.array(feats, dtype=np.int64),
        "nodes_splits": numpy_helper.from_array(np.array(splits, dtype=dt), "splits"),
        # 0 encodes BRANCH_LEQ ("x <= split" takes the true branch), which
        # matches sklearn's children_left/children_right semantics.
        "nodes_modes": numpy_helper.from_array(
            np.zeros(len(feats), dtype=np.uint8), "modes"
        ),
        "nodes_truenodeids": np.array(true_ids, dtype=np.int64),
        "nodes_trueleafs": np.array(true_leafs, dtype=np.int64),
        "nodes_falsenodeids": np.array(false_ids, dtype=np.int64),
        "nodes_falseleafs": np.array(false_leafs, dtype=np.int64),
        "nodes_missing_value_tracks_true": np.zeros(len(feats), dtype=np.int64),
        "leaf_weights": numpy_helper.from_array(np.array(leaf_w, dtype=dt), "leaf_w"),
        "leaf_targetids": np.array(leaf_t, dtype=np.int64),
        "n_targets": max(1, n_targets),
        "aggregate_function": 1,  # SUM
        "post_transform": 0,  # NONE
    }


def _is_tabm_estimator(estimator) -> bool:
    """Return True when ``estimator`` is a fitted-capable TabM model."""
    try:
        from shinrin.tabm import TabMClassifier, TabMRegressor
    except ImportError:  # pragma: no cover - sklearn missing
        return False
    return isinstance(estimator, (TabMClassifier, TabMRegressor))


def to_onnx(
    estimator,
    X=None,
    feature_names=None,
    class_names=None,
    name="ShinrinTree",
    target_opset=None,
):
    """Convert a fitted shinrin tree, forest, or TabM model to ONNX format.

    Parameters
    ----------
    estimator : fitted tree, forest, or TabM estimator
        The model to export. Must have ``tree_`` (single tree),
        ``estimators_`` (forest) attribute, or be a fitted
        :class:`~shinrin.tabm.TabMClassifier` /
        :class:`~shinrin.tabm.TabMRegressor`.
    X : ndarray of shape (n_samples, n_features), optional
        Training data used to infer input shape and dtype.
        If not provided, defaults to 4 features with float64 dtype.
    feature_names : list of str, optional
        Names of input features. Defaults to ["x0", "x1", ...].
    class_names : list of str, optional
        Class names for classification models.
    name : str
        Name of the ONNX model. Defaults to "ShinrinTree".
    target_opset : int, optional
        ONNX opset version. Defaults to 15. Tree/forest exports require
        ``target_opset >= 15`` and emit nodes from ``ai.onnx.ml`` opset 5
        (the standalone ``TreeEnsemble`` operator), so runtimes must
        support that opset.

    Returns
    -------
    onnx.ModelProto
        The ONNX model representation.

        Tree/forest graphs expose the following outputs (float32 or
        float64, matching the dtype of ``X``):

        - regression: ``predictions`` of shape ``(n_samples,)``
        - classification: ``probabilities`` of shape
          ``(n_samples, n_classes)`` plus integer ``labels``
          (or string labels when ``class_names`` is provided)

        TabM graphs are always float32; see
        :func:`shinrin._tabm_onnx.tabm_to_onnx`.

    Raises
    ------
    ValueError
        If the estimator is not fitted or is not a tree-based model.
    ImportError
        If onnx package is not installed.

    Examples
    --------
    >>> from shinrin import MondrianTreeRegressor
    >>> from shinrin.onnx import to_onnx
    >>> import numpy as np
    >>> X = np.random.randn(100, 4).astype(np.float32)
    >>> y = np.random.randn(100)
    >>> tree = MondrianTreeRegressor(random_state=0)
    >>> tree.fit(X, y)
    >>> onnx_model = to_onnx(tree, X)
    """
    try:
        from onnx import (
            TensorProto,
            helper,
            numpy_helper,
        )
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )

    # Route TabM models to the dedicated exporter (self-contained graph with
    # preprocessing baked in; no ai.onnx.ml ops).
    if _is_tabm_estimator(estimator):
        from shinrin._tabm_onnx import tabm_to_onnx

        return tabm_to_onnx(
            estimator,
            X=X,
            feature_names=feature_names,
            class_names=class_names,
            name=name,
            target_opset=target_opset,
        )

    if target_opset is None:
        target_opset = 15
    if target_opset < 15:
        raise ValueError(
            f"Tree/forest export requires opset >= 15, got {target_opset}"
        )

    # Infer input shape and dtype
    if X is not None:
        X = np.asarray(X)
        n_features = X.shape[1]
        input_dtype = np.dtype(X.dtype).type
    else:
        n_features = 4
        input_dtype = np.float64

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # Collect per-tree info dicts plus estimator-level metadata.
    if hasattr(estimator, "tree_"):
        infos = [_tree_info(estimator)]
        is_gb = False
    elif hasattr(estimator, "estimators_"):
        estimators_attr = estimator.estimators_
        is_2d = hasattr(estimators_attr, "ndim") and estimators_attr.ndim > 1
        first_tree = estimators_attr[0][0] if is_2d else estimators_attr[0]
        if not hasattr(first_tree, "tree_"):
            raise ValueError("Forest estimators must contain tree models.")
        if is_2d:
            # scikit-learn gradient boosting: one regressor tree per stage
            # (the per-class columns are collected separately below).
            infos = [_tree_info(row[0]) for row in estimators_attr]
        else:
            infos = [_tree_info(t) for t in estimators_attr]
        # Plain forests always use a 1D ``estimators_``; 2D marks boosted.
        is_gb = is_2d
    else:
        raise ValueError(
            "Estimator must have a 'tree_' attribute (single tree) or "
            "'estimators_' attribute (forest)."
        )

    n_trees = len(infos)
    learning_rate = float(getattr(estimator, "learning_rate", 1.0))
    classes = getattr(estimator, "classes_", None)
    if classes is not None and np.asarray(classes).size > 1:
        is_regression = False
        n_classes = int(np.asarray(classes).size)
    else:
        n_classes = max(info["n_classes"] for info in infos)
        is_regression = n_classes == 1

    onnx_dtype = _onnx_dtype(input_dtype)
    input_tensor = helper.make_tensor_value_info("X", onnx_dtype, [None, n_features])
    graph_nodes: list = []
    initializers: list = []
    outputs: list = []

    def _add_labels_tail(prob_name: str) -> None:
        graph_nodes.append(
            helper.make_node("ArgMax", [prob_name], ["label_idx"], axis=1, keepdims=0)
        )
        if class_names is not None:
            names = np.array([str(c) for c in class_names], dtype=np.str_)
            initializers.append(numpy_helper.from_array(names, "class_names"))
            graph_nodes.append(
                helper.make_node(
                    "Gather", ["class_names", "label_idx"], ["labels"], axis=0
                )
            )
            outputs.append(
                helper.make_tensor_value_info("labels", TensorProto.STRING, [None])
            )
        else:
            graph_nodes.append(helper.make_node("Identity", ["label_idx"], ["labels"]))
            outputs.append(
                helper.make_tensor_value_info("labels", TensorProto.INT64, [None])
            )

    # Constant base prediction of boosted ensembles (None for forests).
    base_values: np.ndarray | None = None
    if is_gb and hasattr(estimator, "init_"):
        probe_row = (
            np.zeros((1, n_features), dtype=input_dtype)
            if X is None
            else X[:1].astype(input_dtype, copy=False)
        )
        init_pred = np.asarray(estimator.init_.predict(probe_row)).ravel()
        base_values = init_pred.astype(input_dtype)

    def _emit_tree_nodes(
        entry_sets: list[list[tuple[dict, int]]],
        n_targets: int,
        scale: float,
        combine: str,
    ) -> str:
        """One TreeEnsemble node per tree, combined into a single tensor.

        A single node per tree preserves the one-node-per-tree contract
        relied upon by :mod:`shinrin.onnx_import`. Leaf weights are stored
        unscaled unless ``scale`` carries a boosting learning rate (which is
        part of a stage's own semantics). Per-tree outputs merge through an
        Add chain; ``combine="mean"`` divides the sum by the tree count so
        downstream consumers observe true per-tree prediction values.
        """
        outs = []
        for k, entries in enumerate(entry_sets):
            attrs = _ensemble_attributes(entries, n_targets, input_dtype, scale)
            attrs["name"] = f"tree_{k}"
            out = f"tree_out_{k}"
            graph_nodes.append(
                helper.make_node("TreeEnsemble", ["X"], [out], **attrs)
            )
            outs.append(out)
        cur = outs[0]
        for k in range(1, len(outs)):
            nxt = f"tree_add_{k}"
            graph_nodes.append(helper.make_node("Add", [cur, outs[k]], [nxt]))
            cur = nxt
        if combine == "mean" and len(outs) > 1:
            div_name = f"tree_div_{len(outs)}"
            initializers.append(
                numpy_helper.from_array(
                    np.array(float(len(outs)), dtype=input_dtype), div_name
                )
            )
            final = "tree_mean"
            graph_nodes.append(helper.make_node("Div", [cur, div_name], [final]))
            cur = final
        return cur

    if is_regression:
        # Forests average their trees; boosted ensembles sum
        # learning-rate-scaled stages.
        raw = _emit_tree_nodes(
            [[(info, 0)] for info in infos],
            1,
            learning_rate if is_gb else 1.0,
            "sum" if is_gb else "mean",
        )
        if base_values is not None:
            initializers.append(numpy_helper.from_array(base_values, "base"))
            graph_nodes.append(helper.make_node("Add", [raw, "base"], ["offset"]))
            raw = "offset"
        initializers.append(
            numpy_helper.from_array(np.array([1], dtype=np.int64), "ax_last")
        )
        graph_nodes.append(helper.make_node("Squeeze", [raw, "ax_last"], ["predictions"]))
        outputs.append(helper.make_tensor_value_info("predictions", onnx_dtype, [None]))
        task, encoding = "regression", _ENCODING_MEAN

    elif is_gb:
        # Boosted classifiers emit per-class log-odds columns.
        encoding = _ENCODING_LOGIT
        if n_classes == 2:
            raw = _emit_tree_nodes(
                [[(info, 0)] for info in infos], 1, learning_rate, "sum"
            )
            if base_values is not None:
                initializers.append(numpy_helper.from_array(base_values[:1], "base"))
                graph_nodes.append(helper.make_node("Add", [raw, "base"], ["offset"]))
                raw = "offset"
            graph_nodes.append(helper.make_node("Sigmoid", [raw], ["p1"]))
            one = numpy_helper.from_array(np.ones(1, dtype=input_dtype), "one")
            initializers.append(one)
            graph_nodes.append(helper.make_node("Sub", [one, "p1"], ["p0"]))
            graph_nodes.append(
                helper.make_node("Concat", ["p0", "p1"], ["probabilities"], axis=1)
            )
        else:
            # One node per boosting stage; each stage carries n_classes
            # trees whose log-odds feed distinct targets.
            stage_entries = [
                [(_tree_info(tree), c) for c, tree in enumerate(row)]
                for row in estimator.estimators_
            ]
            raw = _emit_tree_nodes(stage_entries, n_classes, learning_rate, "sum")
            if base_values is not None:
                initializers.append(numpy_helper.from_array(base_values, "base"))
                graph_nodes.append(helper.make_node("Add", [raw, "base"], ["offset"]))
                raw = "offset"
            graph_nodes.append(
                helper.make_node("Softmax", [raw], ["probabilities"], axis=1)
            )
        outputs.append(
            helper.make_tensor_value_info("probabilities", onnx_dtype, [None, n_classes])
        )
        task = "classification-logits"
        _add_labels_tail("probabilities")

    else:
        # Plain trees/forests carry normalized class probabilities at their
        # leaves; every tree is replicated once per class so each leaf
        # contributes exactly one (probability, target) tuple.
        prob_out = _emit_tree_nodes(
            [[(info, c) for c in range(n_classes)] for info in infos],
            n_classes,
            1.0,
            "mean",
        )
        graph_nodes.append(helper.make_node("Identity", [prob_out], ["probabilities"]))
        outputs.append(
            helper.make_tensor_value_info("probabilities", onnx_dtype, [None, n_classes])
        )
        task, encoding = "classification", _ENCODING_PROBS
        _add_labels_tail("probabilities")

    graph = helper.make_graph(
        graph_nodes,
        f"{name}_graph",
        [input_tensor],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", target_opset),
            helper.make_opsetid("ai.onnx.ml", TREE_ENSEMBLE_OPSET),
        ],
        producer_name=name,
        ir_version=8,
    )
    props = {
        _TASK_PROP: task,
        _ENCODING_PROP: encoding,
    }
    if base_values is not None:
        props[_BASE_VALUES_PROP] = ",".join(repr(float(v)) for v in base_values)
    helper.set_model_props(model, props)
    return model


def _onnx_dtype(dtype):
    """Map a numpy dtype (class or instance) to an ONNX TensorProto dtype."""
    try:
        from onnx import TensorProto
    except ImportError:  # pragma: no cover
        TensorProto = None

    # Normalize through np.dtype so both ``np.float32`` and
    # ``np.dtype("float32")`` resolve to the same entry.
    mapping: dict[str, Any] = {
        "float16": "FLOAT16",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "int8": "INT8",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
        "uint8": "UINT8",
        "uint16": "UINT16",
        "uint32": "UINT32",
        "uint64": "UINT64",
        "bool": "BOOL",
    }
    type_name = mapping.get(np.dtype(dtype).name, "FLOAT")
    if TensorProto is not None:
        return getattr(TensorProto, type_name)
    return type_name


def save_onnx(estimator, path, X=None, feature_names=None, class_names=None):
    """Save a fitted shinrin model to an ONNX file.

    Parameters
    ----------
    estimator : fitted tree, forest, or TabM estimator
        The model to export.
    path : str
        File path to save the ONNX model.
    X : ndarray, optional
        Training data for shape inference.
    feature_names : list of str, optional
        Feature names.
    class_names : list of str, optional
        Class names for classification.
    """
    model = to_onnx(
        estimator,
        X=X,
        feature_names=feature_names,
        class_names=class_names,
    )
    model_str = model.SerializeToString()
    with open(path, "wb") as f:
        f.write(model_str)
