"""ONNX model importer for shinrin Mondrian trees and forests.

This module provides functionality to import fitted gradient boosting or
random forest models (from ONNX format or scikit-learn) into Mondrian tree
structures, enabling incremental training via partial_fit.

The conversion preserves the original model's predictions while rebuilding
the Mondrian-specific statistics (bounds, tau values, node samples) needed
for the Mondrian process.

Usage
-----
>>> from shinrin import MondrianForestRegressor
>>> import numpy as np
>>> # X_train should be at least ~300 rows for meaningful statistics
>>> model = MondrianForestRegressor.from_model(
...     onnx_model, X_train, y_train
... )
>>> # Continue training with new data
>>> model.partial_fit(X_new, y_new)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from onnx import ModelProto

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SklearnModel = Any  # fitted sklearn tree/forest estimator
OnnxModel = "ModelProto"  # onnx.ModelProto
ModelInput = Union[SklearnModel, "ModelProto"]

# ---------------------------------------------------------------------------
# Minimum sample threshold
# ---------------------------------------------------------------------------

_MIN_SAMPLES_FOR_STATS = 300


# ===================================================================
# ONNX tree extraction
# ===================================================================


def _attr_ints(attrs: dict, name: str) -> np.ndarray:
    return np.array(attrs[name].ints, dtype=np.int64)


def _attr_tensor(attrs: dict, name: str) -> np.ndarray:
    """Read a tensor-typed node attribute into numpy."""
    from onnx import numpy_helper

    return numpy_helper.to_array(attrs[name].t)


def _model_props(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def _parse_v5_attrs(attrs: dict) -> dict:
    """Collect ``ai.onnx.ml.TreeEnsemble`` (opset >= 5) attributes."""
    return {
        "roots": _attr_ints(attrs, "tree_roots"),
        "feature": _attr_ints(attrs, "nodes_featureids"),
        "threshold": _attr_tensor(attrs, "nodes_splits").astype(np.float64),
        "true_ids": _attr_ints(attrs, "nodes_truenodeids"),
        "true_leafs": _attr_ints(attrs, "nodes_trueleafs"),
        "false_ids": _attr_ints(attrs, "nodes_falsenodeids"),
        "false_leafs": _attr_ints(attrs, "nodes_falseleafs"),
        "leaf_weights": _attr_tensor(attrs, "leaf_weights").astype(np.float64),
        "leaf_targetids": _attr_ints(attrs, "leaf_targetids"),
    }


def _walk_v5_parallel(
    parsed: dict,
    roots: list[int],
) -> tuple[list[tuple[int, float]], list[Any], dict[int, tuple[int, int]]]:
    """Walk all per-class copies of one tree simultaneously.

    Copies of the same tree share their split structure; at each recursion
    step the current node ids (one per copy) occupy corresponding
    positions. Leaves collect ``{target: weight}`` maps.

    Returns ``(topology, values, children)`` where ``topology`` holds
    pre-order ``(feature, threshold)`` tuples (-2 marks leaves), ``values``
    holds either ``None`` (interior) or a target->weight dict per node, and
    ``children`` maps interior positions to ``(true_ref, false_ref)``.
    """
    topology: list[tuple[int, float]] = []
    # entries are None for interior nodes, {target: weight} for leaves
    values: list[Any] = []
    children: dict[int, tuple[int, int]] = {}

    def visit(nodes: tuple[int, ...], is_leaf: bool) -> int:
        nid0 = nodes[0]
        idx = len(topology)
        if is_leaf:
            weights: dict[int, float] = {}
            for nid in nodes:
                w = float(parsed["leaf_weights"][nid])
                t = int(parsed["leaf_targetids"][nid])
                if t in weights and weights[t] != w:
                    raise ValueError("Inconsistent leaf weights across class copies")
                weights[t] = w
            topology.append((-2, 0.0))
            values.append(weights)
            return idx
        feat0 = int(parsed["feature"][nid0])
        thr0 = float(parsed["threshold"][nid0])
        tl0 = bool(parsed["true_leafs"][nid0])
        fl0 = bool(parsed["false_leafs"][nid0])
        for nid in nodes[1:]:
            if (
                int(parsed["feature"][nid]) != feat0
                or float(parsed["threshold"][nid]) != thr0
                or bool(parsed["true_leafs"][nid]) != tl0
                or bool(parsed["false_leafs"][nid]) != fl0
            ):
                raise ValueError(
                    "Per-class tree copies differ in structure; cannot merge"
                )
        topology.append((feat0, thr0))
        values.append(None)
        true_ref = visit(tuple(int(parsed["true_ids"][n]) for n in nodes), tl0)
        false_ref = visit(tuple(int(parsed["false_ids"][n]) for n in nodes), fl0)
        children[idx] = (true_ref, false_ref)
        return idx

    visit(tuple(roots), False)
    return topology, values, children


def _extract_v5_tree(parsed: dict, task: str | None) -> tuple[dict, str]:
    """Rebuild a sklearn-contract tree-info dict from parsed v5 attributes.

    Returns ``(tree_info, encoding)`` where encoding is ``"means"``
    (regression), ``"probs"`` (plain classification) or ``"logits"``
    (boosted classification).
    """
    topology, values, children = _walk_v5_parallel(
        parsed, [int(r) for r in parsed["roots"]]
    )

    n_nodes = len(topology)
    feature = np.array([e[0] for e in topology], dtype=np.int64)
    threshold = np.array([e[1] for e in topology], dtype=np.float64)
    left = np.full(n_nodes, -1, dtype=np.int64)
    right = np.full(n_nodes, -1, dtype=np.int64)
    for idx, (t_ref, f_ref) in children.items():
        left[idx] = t_ref
        right[idx] = f_ref

    leaf_rows = [i for i, row in enumerate(values) if row is not None]
    targets = sorted({t for i in leaf_rows for t in values[i]})

    if task == "regression":
        value = np.zeros(n_nodes, dtype=np.float64)
        for i in leaf_rows:
            for w in values[i].values():
                value[i] += w
        info = {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "value": value,
            "n_classes": np.array([1]),
        }
        return info, "means"

    if task == "classification-logits":
        # Boosted classifiers store per-class log-odds columns.
        if len(targets) <= 1:
            col = np.zeros((n_nodes, 1), dtype=np.float64)
            for i in leaf_rows:
                for w in values[i].values():
                    col[i, 0] += w
            info = {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": col,
                "n_classes": np.array([2]),
            }
            return info, "logits"
        n_classes = max(targets) + 1
        mat = np.zeros((n_nodes, n_classes), dtype=np.float64)
        for i in leaf_rows:
            for t, w in values[i].items():
                mat[i, t] = w
        info = {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "value": mat,
            "n_classes": np.array([n_classes]),
        }
        return info, "logits"

    # Plain classification exports: normalized class probabilities.
    n_classes = max(max(targets) + 1, 1)
    mat = np.zeros((n_nodes, n_classes), dtype=np.float64)
    for i in leaf_rows:
        for t, w in values[i].items():
            mat[i, t] = w
    info = {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": mat,
        "n_classes": np.array([n_classes]),
    }
    return info, "probs"


def _extract_tree_from_onnx_node(onnx_model: ModelProto, tree_id: int) -> dict:
    """Extract a single tree's structure from an ONNX model's TreeEnsemble node.

    Supports both the current export format (standalone
    ``ai.onnx.ml.TreeEnsemble`` opset >= 5 attributes) and the legacy
    dialect kept for backwards compatibility with older exports.

    Parameters
    ----------
    onnx_model : ModelProto
        The ONNX model containing TreeEnsemble nodes.
    tree_id : int
        The ID of the tree to extract (one TreeEnsemble node per tree).

    Returns
    -------
    dict
        Dictionary with keys: feature, threshold, left_child, right_child,
        value, n_classes plus an ``encoding`` key describing how ``value``
        should be interpreted (``"means"``, ``"probs"``, ``"logits"`` or
        ``None`` for legacy regression-style extraction).
    """
    tree_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "TreeEnsemble"
    ]
    if tree_id >= len(tree_nodes):
        raise ValueError(
            f"Tree with id {tree_id} not found in ONNX model (found {len(tree_nodes)} trees)"
        )

    attrs = {}
    for attr in tree_nodes[tree_id].attribute:
        attrs[attr.name] = attr

    task = _model_props(onnx_model).get("shinrin_task")

    if "nodes_splits" in attrs:
        # Current format: standalone TreeEnsemble opset >= 5 attributes.
        parsed = _parse_v5_attrs(attrs)
        info, encoding = _extract_v5_tree(parsed, task)
        # Normalize to the sklearn-contract key names used downstream.
        info["left_child"] = info.pop("left")
        info["right_child"] = info.pop("right")
        info["encoding"] = encoding
        return info

    # ------------------------------------------------------------------
    # Legacy dialect (pre-v5 exports).
    # ------------------------------------------------------------------
    feature = np.array(attrs["nodes_feature_ids"].ints, dtype=np.int64)
    threshold = np.array(attrs["nodes_values"].floats, dtype=np.float64)
    left_child = np.array(attrs["nodes_truenode_ids"].ints, dtype=np.int64)
    right_child = np.array(attrs["nodes_falsenode_ids"].ints, dtype=np.int64)

    post_transform = (
        attrs["post_transform"].s.decode() if "post_transform" in attrs else "none"
    )
    is_classification = post_transform in ("sigmoid", "softmax")

    nodes_values = np.array(attrs["nodes_values"].floats, dtype=np.float64)

    # Separate thresholds (internal nodes) and leaf values (leaf nodes).
    threshold = np.where(feature == -2, -2.0, nodes_values)
    value = np.where(feature == -2, nodes_values, 0.0)

    encoding = None
    if is_classification:
        base_values = np.array(attrs["base_values"].floats, dtype=np.float64)

        if post_transform == "sigmoid":
            n_classes = 2
        elif "class_labels" in attrs:
            n_classes = len(attrs["class_labels"].ints)
        else:
            n_classes = max(len(base_values), 1)

        if post_transform == "sigmoid":
            value = value.reshape(-1, 1)
        encoding = "log"

        n_classes_arr = np.array([n_classes])
    else:
        n_classes_arr = np.array([1])

    return {
        "feature": feature,
        "threshold": threshold,
        "left_child": left_child,
        "right_child": right_child,
        "value": value,
        "n_classes": n_classes_arr,
        "encoding": encoding,
    }


# Alias for backward compatibility
_extract_tree_from_onnx = _extract_tree_from_onnx_node


def _onnx_tensor_to_numpy(attr) -> np.ndarray:
    """Convert an ONNX attribute tensor to a numpy array."""
    from onnx import numpy_helper

    if hasattr(attr, "t"):
        t = attr.t
        # If data_type is UNDEFINED (0), try to use int32_data or float_data
        if t.data_type == 0:
            if len(t.int32_data) > 0:
                return np.array(t.int32_data, dtype=np.int64)
            elif len(t.float_data) > 0:
                return np.array(t.float_data, dtype=np.float64)
            elif len(t.int64_data) > 0:
                return np.array(t.int64_data, dtype=np.int64)
            # If tensor is empty, fall through to check attr.ints/floats
        else:
            return numpy_helper.to_array(t)
    # Fall back to attribute-level data
    if hasattr(attr, "ints") and len(attr.ints) > 0:
        return np.array(list(attr.ints), dtype=np.int64)
    elif hasattr(attr, "floats") and len(attr.floats) > 0:
        return np.array(list(attr.floats), dtype=np.float64)
    elif hasattr(attr, "strings") and len(attr.strings) > 0:
        return np.array(list(attr.strings), dtype=object)
    else:
        raise ValueError(f"Cannot convert ONNX attribute of type {type(attr)}")


# ===================================================================
# Sklearn model inspection
# ===================================================================


def _is_sklearn_forest(model: SklearnModel) -> bool:
    """Check if a sklearn model is a forest (has estimators_)."""
    return hasattr(model, "estimators_")


def _is_sklearn_tree(model: SklearnModel) -> bool:
    """Check if a sklearn model is a single tree (has tree_)."""
    return hasattr(model, "tree_")


def _get_sklearn_tree_info(model: SklearnModel) -> dict:
    """Extract tree structure from a sklearn tree model.

    Parameters
    ----------
    model : sklearn tree estimator
        Must have a ``tree_`` attribute.

    Returns
    -------
    dict
        Dictionary with keys: feature, threshold, left_child, right_child,
        value, n_classes, n_features.
    """
    t = model.tree_
    n_classes = np.array([int(t.n_classes[0])])

    feature = t.feature.copy().astype(np.int64)
    threshold = t.threshold.copy().astype(np.float64)
    left_child = t.children_left.copy().astype(np.int64)
    right_child = t.children_right.copy().astype(np.int64)

    # Extract node values
    raw_value = (
        t.value
    )  # shape: (n_nodes, n_outputs, max_n_classes) or (n_nodes, max_n_classes)

    if n_classes[0] == 1:
        # Regression
        if raw_value.ndim == 3:
            value = raw_value[:, 0, 0].astype(np.float64)
        else:
            value = raw_value.ravel().astype(np.float64)
    else:
        # Classification: store class counts
        if raw_value.ndim == 3:
            value = raw_value[:, 0, : n_classes[0]].astype(np.float64)
        else:
            value = raw_value[:, : n_classes[0]].astype(np.float64)

    return {
        "feature": feature,
        "threshold": threshold,
        "left_child": left_child,
        "right_child": right_child,
        "value": value,
        "n_classes": n_classes,
        "n_features": int(t.n_features),
    }


# ===================================================================
# Mondrian tree conversion
# ===================================================================


def _compute_bounds_for_node(
    tree_info: dict,
    node_id: int,
    n_features: int,
) -> tuple[list[float], list[float]]:
    """Compute the bounding box for a node by traversing from root.

    The bounds are derived from the split thresholds: each split narrows
    the feature range based on which side of the threshold the node is.

    Parameters
    ----------
    tree_info : dict
        Tree structure with feature, threshold, left_child, right_child.
    node_id : int
        The node index to compute bounds for.
    n_features : int
        Number of features.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) each of length n_features.
    """
    # Start with infinite bounds
    lower_bounds = [float("-inf")] * n_features
    upper_bounds = [float("inf")] * n_features

    # Traverse from root to the target node
    current = 0  # root is always node 0
    while current != node_id:
        feat = tree_info["feature"][current]
        thresh = tree_info["threshold"][current]
        left = tree_info["left_child"][current]
        right = tree_info["right_child"][current]

        if feat >= 0:  # Internal node
            if current == left:
                # Going left: upper bound for this feature becomes threshold
                upper_bounds[feat] = min(upper_bounds[feat], float(thresh))
                current = left
            elif current == right:
                # Going right: lower bound for this feature becomes threshold
                lower_bounds[feat] = max(lower_bounds[feat], float(thresh))
                current = right
            else:
                # Should not happen in a valid tree
                break
        else:
            break

    return lower_bounds, upper_bounds


def _compute_tau(
    tree_info: dict,
    node_id: int,
    n_features: int,
) -> float:
    """Compute the tau (split time) for a node.

    Tau is computed as the exponential rate parameter, estimated from the
    feature space extent at the split point. For converted trees, we use
    a simple heuristic based on tree depth and feature range.

    Parameters
    ----------
    tree_info : dict
        Tree structure.
    node_id : int
        The node index.
    n_features : int
        Number of features.

    Returns
    -------
    float
        The tau value for this node.
    """
    # Count depth by traversing from root
    depth = 0
    current = 0
    path = [0]
    while current != node_id:
        left = tree_info["left_child"][current]
        right = tree_info["right_child"][current]
        if node_id in (left, right):
            depth += 1
            current = left if node_id == left else right
            path.append(current)
        else:
            break

    # Use depth-based heuristic for tau
    # Deeper nodes have smaller tau (later splits)
    # Root gets tau = 1.0, each level divides by 2
    if depth == 0:
        return 1.0

    tau = 1.0 / (2**depth)
    return tau


def _compute_node_samples(
    tree_info: dict,
    node_id: int,
    X: np.ndarray,
) -> float:
    """Compute the number of samples that would reach a node.

    For converted models, we count how many training samples would fall
    into each node based on the tree's split structure.

    Parameters
    ----------
    tree_info : dict
        Tree structure.
    node_id : int
        The node index.
    X : np.ndarray
        Training data of shape (n_samples, n_features).

    Returns
    -------
    float
        Number of samples reaching this node.
    """
    n_samples = X.shape[0]
    node_count = len(tree_info["feature"])
    counts = np.zeros(node_count, dtype=np.float64)

    # Root gets all samples
    counts[0] = float(n_samples)

    # Traverse and propagate counts (nodes should be in topological order)
    for i in range(node_count):
        if counts[i] == 0:
            continue  # No samples reach this node
        feat = tree_info["feature"][i]
        if feat < 0:  # Leaf node
            continue
        thresh = tree_info["threshold"][i]
        left = tree_info["left_child"][i]
        right = tree_info["right_child"][i]

        if left >= 0:
            mask_left = X[:, feat] <= thresh
            counts[left] += np.sum(mask_left)
        if right >= 0:
            mask_right = X[:, feat] > thresh
            counts[right] += np.sum(mask_right)

    return float(max(counts[node_id], 1.0))


def _convert_sklearn_tree_to_mondrian(
    tree_info: dict,
    X: np.ndarray,
    n_features: int,
    from_onnx: bool | str = False,
) -> dict:
    """Convert a sklearn tree structure to Mondrian tree format.

    Parameters
    ----------
    tree_info : dict
        Tree structure from sklearn model or ONNX extraction.
    X : np.ndarray
        Training data for computing node samples.
    n_features : int
        Number of features.
    from_onnx : bool or str
        Encoding of ``tree_info["value"]`` when the structure originates
        from an ONNX export: ``"probs"`` for per-leaf normalized class
        probabilities, ``"logits"`` for log-odds columns (converted to
        pseudo-counts), and any truthy value for the legacy log-prob
        interpretation. ``False`` treats values as sklearn class counts.

    Returns
    -------
    dict
        Mondrian tree data with keys: left_child, right_child, feature,
        threshold, n_node_samples, mean, value, lower_bounds, upper_bounds,
        tau.
    """
    n_nodes = len(tree_info["feature"])
    n_classes = int(tree_info["n_classes"][0])
    is_regression = n_classes == 1

    # Initialize arrays
    left_child = tree_info["left_child"].copy()
    right_child = tree_info["right_child"].copy()
    feature = tree_info["feature"].copy()
    threshold = tree_info["threshold"].copy()

    # Compute bounds, tau, and node samples for each node
    n_node_samples = np.zeros(n_nodes, dtype=np.int64)
    lower_bounds = np.zeros((n_nodes, n_features), dtype=np.float32)
    upper_bounds = np.zeros((n_nodes, n_features), dtype=np.float32)
    tau = np.zeros(n_nodes, dtype=np.float32)

    for i in range(n_nodes):
        lb, ub = _compute_bounds_for_node(tree_info, i, n_features)
        lower_bounds[i] = lb
        upper_bounds[i] = ub
        tau[i] = _compute_tau(tree_info, i, n_features)
        n_node_samples[i] = _compute_node_samples(tree_info, i, X)

    # Compute mean and value arrays
    raw_value = tree_info["value"]

    if is_regression:
        # For regression, ONNX only stores leaf values; internal nodes are zero.
        # Propagate leaf values to internal nodes by weighted averaging based on
        # n_node_samples (weighted mean of children's values).
        n_nodes = len(tree_info["feature"])
        filled_value = raw_value.copy().astype(np.float64).ravel()
        left_child = tree_info["left_child"]
        right_child = tree_info["right_child"]
        # Process nodes in reverse order so children are always processed first
        for i in range(n_nodes - 1, -1, -1):
            if left_child[i] < 0:  # leaf node
                continue
            left = int(left_child[i])
            right = int(right_child[i])
            left_samples = n_node_samples[left]
            right_samples = n_node_samples[right]
            total_samples = left_samples + right_samples
            if total_samples == 0:
                filled_value[i] = (filled_value[left] + filled_value[right]) / 2.0
            else:
                filled_value[i] = (
                    left_samples * filled_value[left]
                    + right_samples * filled_value[right]
                ) / total_samples

        mean = filled_value.copy()
        value = filled_value.copy()
    else:
        # For classification, value stores class counts (sklearn) or
        # probabilities / log-odds / log-probs (ONNX). Convert as needed.
        if from_onnx == "probs":
            # Current exports: per-leaf normalized class probabilities.
            prob = raw_value.reshape(-1, n_classes).copy().astype(np.float64)
            total = n_node_samples.reshape(-1, 1)
            total[total == 0] = 1.0
            value = (prob * total).ravel()
            mean = prob.ravel()
        elif from_onnx:
            # Legacy ONNX stores log-odds (binary) or log-probs (multi-class)
            # Convert to pseudo-counts using n_node_samples
            logit_or_logprob = raw_value.copy().astype(np.float64)
            if logit_or_logprob.ndim == 1:
                logit_or_logprob = logit_or_logprob.reshape(-1, 1)
            if n_classes == 2 and logit_or_logprob.shape[1] == 1:
                # Binary: logit shape (n_nodes, 1)
                prob = 1.0 / (1.0 + np.exp(-logit_or_logprob))
                # Pseudo-counts: [count_0, count_1]
                total = n_node_samples.reshape(-1, 1)
                total[total == 0] = 1.0
                count_1 = (prob * total).ravel()
                count_0 = ((1.0 - prob) * total).ravel()
                value = np.column_stack([count_0, count_1]).ravel()
            else:
                # Multi-class: log-probs shape (n_nodes, n_classes)
                prob = np.exp(logit_or_logprob)
                prob = prob / prob.sum(axis=1, keepdims=True)
                total = n_node_samples.reshape(-1, 1)
                total[total == 0] = 1.0
                value = (prob * total).ravel()
            # Compute mean as predicted class probability
            total_samples = n_node_samples.reshape(-1, 1)
            total_samples[total_samples == 0] = 1.0
            mean = (value.reshape(-1, n_classes) / total_samples).ravel()
        else:
            # sklearn: value stores class counts
            value = raw_value.copy().astype(np.float64).ravel()
            # Compute mean as predicted class probability
            total = n_node_samples.reshape(-1, 1)
            total[total == 0] = 1.0
            mean = (value / total).astype(np.float64)

    return {
        "left_child": left_child,
        "right_child": right_child,
        "feature": feature,
        "threshold": threshold,
        "n_node_samples": n_node_samples,
        "mean": mean,
        "value": value,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "tau": tau,
    }


def _convert_onnx_tree_to_mondrian(
    onnx_model: ModelProto,
    tree_id: int,
    tree_info: dict,
    X: np.ndarray,
) -> dict:
    """Convert an ONNX tree to Mondrian tree format.

    Parameters
    ----------
    onnx_model : ModelProto
        The ONNX model.
    tree_id : int
        The tree ID in the ONNX model.
    tree_info : dict
        Extracted tree structure from ONNX (includes an ``encoding`` key).
    X : np.ndarray
        Training data for computing node samples.

    Returns
    -------
    dict
        Mondrian tree data.
    """
    n_features = X.shape[1]
    encoding = tree_info.get("encoding")
    if encoding is None:
        encoding = True
    return _convert_sklearn_tree_to_mondrian(
        tree_info, X, n_features, from_onnx=encoding
    )


# ===================================================================
# Public API: from_model
# ===================================================================


def _is_onnx_model(model: Any) -> bool:
    """Check if an object is an ONNX ModelProto."""
    return hasattr(model, "graph") and hasattr(model, "producer_name")


def _count_trees_in_onnx(onnx_model: ModelProto) -> int:
    """Count the number of trees in an ONNX model."""
    # Count TreeEnsemble nodes (each node represents one tree)
    return sum(1 for node in onnx_model.graph.node if node.op_type == "TreeEnsemble")


def from_model(
    model: ModelInput,
    X: np.ndarray,
    y: np.ndarray,
    cls: type,
) -> Any:
    """Create a Mondrian model from a fitted sklearn or ONNX model.

    This function converts a fitted gradient boosting or random forest model
    (provided as a sklearn estimator or ONNX model) into a Mondrian tree or
    forest. The conversion preserves the original model's predictions while
    rebuilding Mondrian-specific statistics (bounds, tau values, node samples)
    needed for incremental training via partial_fit.

    Parameters
    ----------
    model : sklearn estimator or ONNX ModelProto
        A fitted sklearn tree/forest estimator (with ``tree_`` or
        ``estimators_`` attribute) or an ONNX model proto.
    X : array-like of shape (n_samples, n_features)
        Training data used to compute Mondrian statistics. Should have at
        least 300 samples for meaningful statistics.
    y : array-like of shape (n_samples,)
        Target values.
    cls : type
        The Mondrian class to instantiate (e.g., MondrianTreeRegressor,
        MondrianForestRegressor).

    Returns
    -------
    Mondrian tree or forest instance
        A fitted Mondrian model that produces the same predictions as the
        original model and supports partial_fit.

    Raises
    ------
    ValueError
        If the model is not a valid tree or forest model, or if X and y
        have incompatible shapes.
    ImportError
        If onnx is required but not installed.

    Warnings
    --------
    UserWarning
        Issued if the training dataset has fewer than 300 samples, as the
        computed statistics may not be representative.

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from shinrin import MondrianForestRegressor
    >>> from shinrin.onnx import to_onnx, save_onnx
    >>> import tempfile
    >>>
    >>> # Train a sklearn model
    >>> sklearn_model = GradientBoostingRegressor(n_estimators=5, max_depth=3)
    >>> sklearn_model.fit(X_train, y_train)
    >>>
    >>> # Export to ONNX
    >>> onnx_model = to_onnx(sklearn_model, X_train)
    >>>
    >>> # Convert to Mondrian forest (one-step)
    >>> mondrian_model = MondrianForestRegressor.from_model(
    ...     onnx_model, X_train, y_train
    ... )
    >>>
    >>> # Continue training with new data
    >>> mondrian_model.partial_fit(X_new, y_new)
    """
    from shinrin._compat.sklearn_utils_validation import (
        check_X_y,
    )

    X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)

    n_samples = X.shape[0]

    # Warn about small datasets
    if n_samples < _MIN_SAMPLES_FOR_STATS:
        warnings.warn(
            f"Training dataset has only {n_samples} samples. For meaningful "
            f"Mondrian statistics, at least {_MIN_SAMPLES_FOR_STATS} samples "
            "are recommended. The converted model will still work, but "
            "statistics may be less representative.",
            UserWarning,
            stacklevel=4,
        )

    # Only support ONNX models
    if not _is_onnx_model(model):
        raise ValueError(
            "Model must be an ONNX ModelProto. Use shinrin.onnx.to_onnx() "
            "to convert sklearn models first."
        )

    # Determine if this is a forest or single tree
    n_trees = _count_trees_in_onnx(model)
    is_forest = n_trees > 1

    if is_forest:
        return _from_model_forest(model, X, y, cls)
    else:
        return _from_model_single_tree(model, X, y, cls)


def _from_model_single_tree(
    model: ModelInput,
    X: np.ndarray,
    y: np.ndarray,
    cls: type,
) -> Any:
    """Convert a single tree model to Mondrian format."""
    # Import the native tree builder (active backend)
    from shinrin._backend import get_backend_module

    Tree = get_backend_module().Tree

    # Extract tree structure from ONNX
    tree_info = _extract_tree_from_onnx_node(model, 0)
    n_classes = tree_info["n_classes"]
    tree_data = _convert_onnx_tree_to_mondrian(model, 0, tree_info, X)

    # Create the Mondrian tree
    n_features = X.shape[1]
    n_classes_arr = np.array(n_classes, dtype=np.intp)
    tree = Tree(n_features, n_classes_arr, 1)

    # Populate tree from arrays using the native method
    tree.populate_from_arrays(
        left_child=tree_data["left_child"],
        right_child=tree_data["right_child"],
        feature=tree_data["feature"],
        threshold=tree_data["threshold"],
        n_node_samples=tree_data["n_node_samples"],
        value=tree_data["value"],
        tau=tree_data["tau"],
        lower_bounds=tree_data["lower_bounds"],
        upper_bounds=tree_data["upper_bounds"],
    )

    # Extract classes from sklearn model if available
    classes = None
    if _is_sklearn_tree(model) and hasattr(model, "classes_"):
        classes = model.classes_

    # Create the Mondrian tree wrapper
    if "Regressor" in cls.__name__:
        from shinrin._skgarden.mondrian.tree.tree import MondrianTreeRegressor

        mondrian_tree = MondrianTreeRegressor.__new__(MondrianTreeRegressor)
        mondrian_tree.tree_ = tree
        mondrian_tree.n_features_ = n_features
        mondrian_tree.n_classes_ = 1
        mondrian_tree.n_outputs_ = 1
        mondrian_tree.max_depth = None
        mondrian_tree.min_samples_split = 2
        mondrian_tree.random_state = None
        mondrian_tree.classes_ = None
        mondrian_tree.first_ = True  # Mark as converted (not fresh fit)
        mondrian_tree._onnx_converted = True
        return mondrian_tree
    else:
        from shinrin._skgarden.mondrian.tree.tree import MondrianTreeClassifier

        mondrian_tree = MondrianTreeClassifier.__new__(MondrianTreeClassifier)
        mondrian_tree.tree_ = tree
        mondrian_tree.n_features_ = n_features
        mondrian_tree.n_classes_ = int(n_classes[0])
        mondrian_tree.n_outputs_ = 1
        mondrian_tree.max_depth = None
        mondrian_tree.min_samples_split = 2
        mondrian_tree.random_state = None
        mondrian_tree.classes_ = (
            classes
            if classes is not None
            else np.array([f"class_{i}" for i in range(int(n_classes[0]))])
        )
        mondrian_tree.first_ = True
        mondrian_tree._onnx_converted = True
        return mondrian_tree


def _from_model_forest(
    model: ModelInput,
    X: np.ndarray,
    y: np.ndarray,
    cls: type,
) -> Any:
    """Convert a forest model to Mondrian format."""
    # Determine the base tree class
    is_classifier = "Classifier" in cls.__name__

    if is_classifier:
        from shinrin._skgarden.mondrian.tree.tree import MondrianTreeClassifier

        base_tree_cls = MondrianTreeClassifier
    else:
        from shinrin._skgarden.mondrian.tree.tree import MondrianTreeRegressor

        base_tree_cls = MondrianTreeRegressor

    # Extract tree count and IDs from ONNX model
    tree_ids = [
        i for i, node in enumerate(model.graph.node) if node.op_type == "TreeEnsemble"
    ]

    # Extract base_values from ONNX model if available (for GradientBoosting).
    # Current exports record them as model metadata; legacy exports carried
    # a base_values attribute on the node.
    base_values: np.ndarray | None = None
    props = _model_props(model)
    if "shinrin_base_values" in props:
        base_values = np.array(
            [float(v) for v in props["shinrin_base_values"].split(",")],
            dtype=np.float64,
        )
    else:
        for node in model.graph.node:
            if node.op_type == "TreeEnsemble":
                for attr in node.attribute:
                    if attr.name == "base_values":
                        base_values = np.array(attr.floats, dtype=np.float64)
                        break
                if base_values is not None:
                    break

    # Convert each tree in parallel
    def _convert_single_tree(tree_id: int) -> Any:
        tree_info = _extract_tree_from_onnx_node(model, tree_id)
        n_classes = tree_info["n_classes"]
        tree_data = _convert_onnx_tree_to_mondrian(model, tree_id, tree_info, X)
        # Add base_values to the first tree for GradientBoosting
        if tree_id == 0 and base_values is not None:
            # Add base_values to all leaf values
            tree_data["value"] = tree_data["value"].copy()
            if is_classifier:
                # For classification, value is raveled (n_nodes * n_classes,)
                # Reshape to (n_nodes, n_classes), add the per-class base
                # log-odds, then ravel again.
                n_classes_val = int(tree_info["n_classes"][0])
                value_2d = tree_data["value"].reshape(-1, n_classes_val)
                if base_values.size == 1:
                    value_2d[:, 0] += base_values[0]
                else:
                    value_2d[:, : base_values.size] += base_values
                tree_data["value"] = value_2d.ravel()
            else:
                # For regression, value is 1D
                tree_data["value"] = tree_data["value"] + base_values[0]

        # Create the native tree (active backend)
        from shinrin._backend import get_backend_module

        Tree = get_backend_module().Tree

        n_features = X.shape[1]
        n_classes_arr = np.array(n_classes, dtype=np.intp)
        tree = Tree(n_features, n_classes_arr, 1)

        tree.populate_from_arrays(
            left_child=tree_data["left_child"],
            right_child=tree_data["right_child"],
            feature=tree_data["feature"],
            threshold=tree_data["threshold"],
            n_node_samples=tree_data["n_node_samples"],
            value=tree_data["value"],
            tau=tree_data["tau"],
            lower_bounds=tree_data["lower_bounds"],
            upper_bounds=tree_data["upper_bounds"],
        )

        # Create the Mondrian tree wrapper
        tree_wrapper = base_tree_cls.__new__(base_tree_cls)
        tree_wrapper.tree_ = tree
        tree_wrapper.n_features_ = n_features
        tree_wrapper.n_classes_ = int(n_classes[0])
        tree_wrapper.n_outputs_ = 1
        tree_wrapper.max_depth = None
        tree_wrapper.min_samples_split = 2
        tree_wrapper.random_state = None
        tree_wrapper.classes_ = np.array(
            [f"class_{i}" for i in range(int(n_classes[0]))]
        )
        tree_wrapper.first_ = True
        tree_wrapper._onnx_converted = True
        return tree_wrapper

    # Convert trees sequentially (parallel fails because PyTree is unsendable)
    trees = [_convert_single_tree(tid) for tid in tree_ids]

    # Create the forest instance
    if is_classifier:
        from shinrin._skgarden.mondrian.ensemble.forest import (
            MondrianForestClassifier,
        )

        forest = MondrianForestClassifier.__new__(MondrianForestClassifier)
        forest.estimators_ = trees
        forest.n_features_ = X.shape[1]
        forest.n_classes_ = forest.estimators_[0].n_classes_
        forest.classes_ = forest.estimators_[0].classes_
        forest.n_estimators = len(trees)
        forest.n_outputs_ = 1
        forest.max_depth = None
        forest.min_samples_split = 2
        forest.bootstrap = False
        forest.n_jobs = -1
        forest.random_state = None
        forest.verbose = 0
    else:
        from shinrin._skgarden.mondrian.ensemble.forest import (
            MondrianForestRegressor,
        )

        forest = MondrianForestRegressor.__new__(MondrianForestRegressor)
        forest.estimators_ = trees
        forest.n_features_ = X.shape[1]
        forest.n_classes_ = 1
        forest.classes_ = None
        forest.n_estimators = len(trees)
        forest.n_outputs_ = 1
        forest.max_depth = None
        forest.min_samples_split = 2
        forest.bootstrap = False
        forest.n_jobs = -1
        forest.random_state = None
        forest.verbose = 0

    forest.first_ = True  # Mark as converted
    forest._onnx_converted = True  # Mark as ONNX/sklearn converted
    # For GradientBoosting, sum predictions instead of averaging
    # base_values is non-zero only for GradientBoosting (mean of y)
    # For RandomForest, base_values is 0.0
    if base_values is not None and bool(np.any(base_values != 0.0)):
        forest._onnx_sum_predictions = True
    return forest
