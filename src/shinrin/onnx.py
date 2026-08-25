"""ONNX exporter for shinrin models.

``to_onnx`` converts a fitted shinrin estimator into a self-contained ONNX
model proto. Supported model families and their export strategies:

- **Trees / forests** (:mod:`shinrin._skgarden`): emitted as ``ai.onnx.ml``
  ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier`` graphs (domain
  version 3). All trees of a forest are packed into a single node whose
  aggregated output equals the average of the per-tree predictions.
- **Quantile trees / forests**: exported exactly for one quantile chosen at
  export time (see :func:`to_onnx`); training targets are embedded in the
  graph so weighted-percentile parity is preserved (forests).
- **Rule-based models** (SkopeRules, CorelsClassifier, OrdtClassifier,
  SPOTClassifier): rules/trees are compiled to standard-domain boolean ops
  or tree ensembles.

All exports use dynamic batch size and expect a float32 ``X`` input tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from onnx import TensorProto
except ImportError:  # pragma: no cover
    TensorProto: Any = None

# ai.onnx.ml domain version used by all tree-ensemble exports. Version 3
# introduced TreeEnsembleRegressor/TreeEnsembleClassifier and is supported
# by every major runtime.
AI_ONNX_ML_OPSET = 3


def _require_onnx():
    try:
        from onnx import TensorProto, helper

        return TensorProto, helper
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )


def _set_props(model, props: dict[str, str]) -> None:
    from onnx import helper

    helper.set_model_props(model, props)


# ---------------------------------------------------------------------------
# Ensemble builder shared by regressor / classifier exports
# ---------------------------------------------------------------------------


class _EnsembleBuilder:
    """Accumulates node arrays for one multi-tree ONNX ensemble."""

    def __init__(self) -> None:
        self.nodes_treeids: list[int] = []
        self.nodes_nodeids: list[int] = []
        self.nodes_featureids: list[int] = []
        self.nodes_modes: list[str] = []
        self.nodes_values: list[float] = []
        self.nodes_truenodeids: list[int] = []
        self.nodes_falsenodeids: list[int] = []
        # Per-tree mapping leaf node id -> value vector (``tree_.value``
        # flattened: mean for regression trees, class counts otherwise).
        self.leaves_by_tree: list[dict[int, np.ndarray]] = []

    def add_tree(self, tree_id: int, tree) -> None:
        """Append one fitted ``tree_`` (sklearn or native Mondrian layout)."""
        left = np.asarray(tree.children_left)
        right = np.asarray(tree.children_right)
        feature = np.asarray(tree.feature)
        threshold = np.asarray(tree.threshold, dtype=np.float64)
        value = np.asarray(tree.value, dtype=np.float64)
        n_nodes = int(tree.node_count)

        leaves: dict[int, np.ndarray] = {}
        for i in range(n_nodes):
            self.nodes_treeids.append(tree_id)
            self.nodes_nodeids.append(i)
            if left[i] < 0 and right[i] < 0:
                self.nodes_featureids.append(0)
                self.nodes_modes.append("LEAF")
                self.nodes_values.append(0.0)
                self.nodes_truenodeids.append(i)
                self.nodes_falsenodeids.append(i)
                leaves[i] = value[i].reshape(-1)
            else:
                self.nodes_featureids.append(int(feature[i]))
                self.nodes_modes.append("BRANCH_LEQ")
                self.nodes_values.append(float(threshold[i]))
                true_id = int(left[i]) if left[i] >= 0 else i
                false_id = int(right[i]) if right[i] >= 0 else i
                self.nodes_truenodeids.append(true_id)
                self.nodes_falsenodeids.append(false_id)
        self.leaves_by_tree.append(leaves)


def _tree_iter(estimator) -> tuple[list[Any], bool]:
    """Return ``(list_of_tree_, is_classification)`` for a tree/forest model."""
    trees_attr = getattr(estimator, "estimators_", None)
    if trees_attr is not None:
        arr = getattr(trees_attr, "ndim", 1)
        if isinstance(arr, int) and arr > 1:
            first = trees_attr[0][0]
            estimators = [e[0] for e in trees_attr]
        else:
            first = trees_attr[0]
            estimators = list(trees_attr)
        if not hasattr(first, "tree_"):
            raise ValueError("Forest estimators must contain tree models.")
        trees = [t.tree_ for t in estimators]
    elif getattr(estimator, "tree_", None) is not None:
        trees = [estimator.tree_]
    else:
        raise ValueError(
            f"Unsupported estimator {type(estimator).__name__}: expected a "
            "fitted tree ('tree_') or forest ('estimators_') model."
        )

    n_classes = int(np.asarray(trees[0].n_classes).ravel()[0])
    return trees, n_classes > 1


# ---------------------------------------------------------------------------
# Generic tree / forest export
# ---------------------------------------------------------------------------


def _generic_to_onnx(
    estimator,
    feature_names: list[str] | None,
    name: str,
    target_opset: int,
):
    """Export a fitted tree or forest to an ai.onnx.ml tree-ensemble graph."""
    TensorProto, helper = _require_onnx()

    trees, is_classification = _tree_iter(estimator)
    n_features_in = int(trees[0].n_features)
    n_classes = (
        max(int(np.asarray(t.n_classes).ravel()[0]) for t in trees)
        if is_classification
        else 1
    )
    n_trees = len(trees)

    # GradientBoosting ensembles sum learning-rate-scaled tree outputs on top
    # of the initial constant prediction; forests plain-average raw outputs.
    is_gb = type(estimator).__name__.startswith("GradientBoosting")
    if is_gb:
        learning_rate = float(getattr(estimator, "learning_rate", 1.0))
        # GB trees are regressors, so ``is_classification`` (derived from
        # per-tree n_classes) cannot detect a boosting classifier; use the
        # estimator's own classes_ instead.
        if is_classification or getattr(estimator, "classes_", None) is not None:
            raise NotImplementedError(
                "ONNX export for GradientBoosting classifiers is not supported; "
                "use GradientBoostingRegressor or a shinrin forest model."
            )
        base_value = float(np.ravel(estimator.init_.constant_)[0])
    else:
        learning_rate = 1.0
        base_value = 0.0

    builder = _EnsembleBuilder()
    for tid, tree in enumerate(trees):
        builder.add_tree(tid, tree)

    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features_in])
    ]
    opset_imports = [
        helper.make_opsetid("", target_opset),
        helper.make_opsetid("ai.onnx.ml", AI_ONNX_ML_OPSET),
    ]

    common_attrs = {
        "nodes_treeids": np.array(builder.nodes_treeids, dtype=np.int64),
        "nodes_nodeids": np.array(builder.nodes_nodeids, dtype=np.int64),
        "nodes_featureids": np.array(builder.nodes_featureids, dtype=np.int64),
        "nodes_modes": builder.nodes_modes,
        "nodes_values": np.array(builder.nodes_values, dtype=np.float32),
        "nodes_truenodeids": np.array(builder.nodes_truenodeids, dtype=np.int64),
        "nodes_falsenodeids": np.array(builder.nodes_falsenodeids, dtype=np.int64),
        "nodes_missing_value_tracks_true": np.zeros(
            len(builder.nodes_modes), dtype=np.int64
        ),
    }

    if not is_classification:
        target_treeids, target_nodeids, target_weights = [], [], []
        for tid, leaves in enumerate(builder.leaves_by_tree):
            for nid, vec in leaves.items():
                target_treeids.append(tid)
                target_nodeids.append(nid)
                target_weights.append(float(vec[0]))
        attrs = {
            **common_attrs,
            "n_targets": 1,
            # GB sums lr-scaled trees; single trees emit one SUM entry;
            # forests average their trees' outputs.
            "aggregate_function": ("AVERAGE" if n_trees > 1 and not is_gb else "SUM"),
            "post_transform": "NONE",
            "target_treeids": np.array(target_treeids, dtype=np.int64),
            "target_nodeids": np.array(target_nodeids, dtype=np.int64),
            "target_ids": np.zeros(len(target_weights), dtype=np.int64),
            "target_weights": np.array(target_weights, dtype=np.float32)
            * np.float32(learning_rate),
        }
        node = helper.make_node(
            "TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["te_out"],
            domain="ai.onnx.ml",
            **attrs,
        )
        nodes = [node]
        initializers = [helper.make_tensor("neg1", TensorProto.INT64, [1], [-1])]
        tail_outputs = [
            helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None])
        ]
        if base_value != 0.0:
            initializers.append(
                helper.make_tensor(
                    "base_value", TensorProto.FLOAT, [], [np.float32(base_value)]
                )
            )
            nodes.append(helper.make_node("Add", ["te_out", "base_value"], ["te_base"]))
            reshape = helper.make_node("Reshape", ["te_base", "neg1"], ["predictions"])
        else:
            reshape = helper.make_node("Reshape", ["te_out", "neg1"], ["predictions"])
        nodes.append(reshape)
        graph = helper.make_graph(
            nodes,
            f"{name}_graph",
            inputs,
            tail_outputs,
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=opset_imports, producer_name=name, ir_version=8
        )
        return model

    # Classification: each leaf contributes its normalized class distribution;
    # summed over trees this yields the averaged probabilities. post_transform
    # NONE keeps them normalized since every tree's vector sums to 1.
    class_treeids, class_nodeids, class_ids, class_weights = [], [], [], []
    scale = 1.0 / n_trees
    for tid, leaves in enumerate(builder.leaves_by_tree):
        for nid, counts in leaves.items():
            total = float(counts.sum())
            probs = (
                counts / total if total > 0 else np.full_like(counts, 1.0 / n_classes)
            )
            for k in range(n_classes):
                class_treeids.append(tid)
                class_nodeids.append(nid)
                class_ids.append(k)
                class_weights.append(float(probs[k]) * scale)

    classes = np.asarray(getattr(estimator, "classes_", np.arange(n_classes)))
    attrs = {
        **common_attrs,
        "class_treeids": np.array(class_treeids, dtype=np.int64),
        "class_nodeids": np.array(class_nodeids, dtype=np.int64),
        "class_ids": np.array(class_ids, dtype=np.int64),
        "class_weights": np.array(class_weights, dtype=np.float32),
        "post_transform": "NONE",
    }
    if classes.shape[0] == n_classes:
        if np.issubdtype(classes.dtype, np.number):
            attrs["classlabels_int64s"] = classes.astype(np.int64)
            labels_dtype = TensorProto.INT64
        else:
            attrs["classlabels_strings"] = np.array([str(c) for c in classes], object)
            labels_dtype = TensorProto.STRING
    else:  # pragma: no cover - classes_ always matches n_classes when fitted
        labels_dtype = TensorProto.INT64

    node = helper.make_node(
        "TreeEnsembleClassifier",
        inputs=["X"],
        outputs=["te_labels", "probabilities"],
        domain="ai.onnx.ml",
        **attrs,
    )
    # ORT derives the ensemble's own label output with the skl2onnx binary
    # convention (score > 0), which disagrees with argmax over plain
    # probability weights. Ignore it and derive labels from the exact
    # averaged probabilities instead.
    if np.issubdtype(classes.dtype, np.number):
        _add_i64_init = helper.make_tensor(
            "classes",
            TensorProto.INT64,
            list(classes.shape),
            classes.astype(np.int64).ravel(),
        )
        labels_dtype = TensorProto.INT64
    else:
        _add_i64_init = helper.make_tensor(
            "classes",
            TensorProto.STRING,
            list(classes.shape),
            np.array([str(c) for c in classes], object).ravel(),
        )
        labels_dtype = TensorProto.STRING
    initializers = [_add_i64_init]
    nodes = [
        node,
        helper.make_node("ArgMax", ["probabilities"], ["amax"], axis=1, keepdims=0),
        helper.make_node("Unsqueeze", ["amax", "cls_ax"], ["amax4g"]),
        helper.make_node("Gather", ["classes", "amax4g"], ["labels2d"], axis=0),
        helper.make_node("Squeeze", ["labels2d", "cls_ax"], ["labels"]),
    ]
    initializers.append(helper.make_tensor("cls_ax", TensorProto.INT64, [1], [1]))
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        inputs,
        [
            helper.make_tensor_value_info("labels", labels_dtype, [None]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, n_classes]
            ),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=opset_imports, producer_name=name, ir_version=8
    )
    return model


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def to_onnx(
    estimator,
    X=None,
    feature_names=None,
    class_names=None,
    name="ShinrinModel",
    target_opset=None,
    quantile=None,
):
    """Convert a fitted shinrin model to ONNX format.

    Parameters
    ----------
    estimator : fitted shinrin estimator
        The model to export. See the module docstring for supported families.
    X : ndarray of shape (n_samples, n_features), optional
        Training-like data used to infer the number of features. The exported
        graph always accepts float32 tensors with a dynamic batch dimension.
    feature_names : list of str, optional
        Names of input features; stored as model metadata.
    class_names : list of str, optional
        Class names for classification models; when provided the ``labels``
        output yields these names instead of integer class values.
    name : str
        Name of the ONNX model.
    target_opset : int, optional
        Default-domain opset version (default 15).
    quantile : int, optional
        For quantile models: quantile (0-100) baked into the exported graph.
        Required for quantile estimators, ignored elsewhere.

    Returns
    -------
    onnx.ModelProto

    Raises
    ------
    ValueError
        If the estimator is not fitted or is not a supported model.
    ImportError
        If the onnx package is not installed.

    Examples
    --------
    >>> from shinrin import MondrianForestRegressor
    >>> from shinrin.onnx import to_onnx
    >>> import numpy as np
    >>> X = np.random.randn(100, 4).astype(np.float32)
    >>> y = np.random.randn(100)
    >>> forest = MondrianForestRegressor(random_state=0).fit(X, y)
    >>> onnx_model = to_onnx(forest, X)
    """
    _require_onnx()

    if X is not None:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got ndim={X.ndim}")
        n_features_in: int | None = X.shape[1]
    else:
        n_features_in = getattr(estimator, "n_features_in_", None)

    if feature_names is None and n_features_in is not None:
        feature_names = [f"x{i}" for i in range(n_features_in)]

    if target_opset is None:
        target_opset = 15

    # --- family routing -------------------------------------------------------
    try:
        from shinrin.mondrian import (
            MondrianForestClassifier,
            MondrianForestRegressor,
            MondrianTreeClassifier,
            MondrianTreeRegressor,
        )

        if isinstance(
            estimator,
            (
                MondrianTreeRegressor,
                MondrianTreeClassifier,
                MondrianForestRegressor,
                MondrianForestClassifier,
            ),
        ):
            from shinrin._mondrian_onnx import mondrian_to_onnx

            return mondrian_to_onnx(
                estimator,
                feature_names=feature_names,
                class_names=class_names,
                name=name,
                target_opset=target_opset,
            )
    except ImportError:  # pragma: no cover - sklearn missing
        pass

    from shinrin._skgarden.quantile.ensemble import BaseForestQuantileRegressor
    from shinrin._skgarden.quantile.tree import BaseTreeQuantileRegressor

    if isinstance(estimator, BaseTreeQuantileRegressor):
        from shinrin._quantile_onnx import quantile_tree_to_onnx

        return quantile_tree_to_onnx(
            estimator,
            quantile=quantile,
            feature_names=feature_names,
            name=name,
            target_opset=target_opset,
        )
    if isinstance(estimator, BaseForestQuantileRegressor):
        from shinrin._quantile_onnx import quantile_forest_to_onnx

        return quantile_forest_to_onnx(
            estimator,
            quantile=quantile,
            feature_names=feature_names,
            name=name,
            target_opset=target_opset,
        )

    try:
        from shinrin._corels.corels import CorelsClassifier

        if isinstance(estimator, CorelsClassifier):
            from shinrin._rules_onnx import corels_to_onnx

            return corels_to_onnx(
                estimator,
                feature_names=feature_names,
                class_names=class_names,
                name=name,
                target_opset=target_opset,
            )
    except ImportError:  # pragma: no cover - sklearn missing
        pass

    try:
        from shinrin._spot.classifier import SPOTClassifier

        if isinstance(estimator, SPOTClassifier):
            from shinrin._rules_onnx import gosdt_to_onnx

            return gosdt_to_onnx(
                estimator,
                feature_names=feature_names,
                class_names=class_names,
                name=name,
                target_opset=target_opset,
            )
    except ImportError:  # pragma: no cover
        pass

    try:
        from shinrin._ordt import OrdtClassifier
        from shinrin._skrules.skope_rules import SkopeRules

        if isinstance(estimator, OrdtClassifier):
            from shinrin._rules_onnx import ordt_to_onnx

            return ordt_to_onnx(
                estimator,
                feature_names=feature_names,
                class_names=class_names,
                name=name,
                target_opset=target_opset,
            )
        if isinstance(estimator, SkopeRules):
            from shinrin._rules_onnx import skope_rules_to_onnx

            return skope_rules_to_onnx(
                estimator,
                feature_names=feature_names,
                name=name,
                target_opset=target_opset,
            )
    except ImportError:  # pragma: no cover - pandas missing
        pass

    model = _generic_to_onnx(estimator, feature_names, name, target_opset)
    props = {
        "shinrin_version": "0.2.0",
        "model_type": type(estimator).__name__,
    }
    if feature_names is not None:
        props["feature_names"] = ",".join(str(f) for f in feature_names)
    _set_props(model, props)
    return model


def save_onnx(estimator, path, X=None, feature_names=None, class_names=None):
    """Save a fitted shinrin model to an ONNX file.

    Parameters
    ----------
    estimator : fitted shinrin estimator
        The model to export.
    path : str
        File path to save the ONNX model.
    X : ndarray, optional
        Training-like data for shape inference.
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
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
