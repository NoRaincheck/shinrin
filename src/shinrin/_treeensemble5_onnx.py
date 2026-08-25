"""ONNX ``ai.onnx.ml`` opset-5 ``TreeEnsemble`` export with ``BRANCH_MEMBER``.

This exporter consumes a fitted tree/forest trained on *target-encoded*
categorical columns plus the :class:`shinrin.TargetEncoder` used during
training, and emits an ``ai.onnx.ml`` (opset 5) ``TreeEnsemble`` where
every encoded-threshold split on a categorical column becomes a
``BRANCH_MEMBER`` node testing raw category-code membership. The exported
graph therefore consumes the **original input feature convention** (raw
category codes; numeric columns untouched) — no target encoder at
inference time.

Attribute encoding notes (validated against onnx 1.22 and onnxruntime
1.29):

- ``nodes_modes`` is a uint8 tensor (``BRANCH_LEQ`` = 0,
  ``BRANCH_MEMBER`` = 6); ``nodes_splits`` / ``membership_values`` /
  ``leaf_weights`` are tensors matching the input dtype. All other
  ``nodes_*`` / ``leaf_*`` fields are plain INTS lists.
- There are no ``nodes_treeids`` / ``nodes_nodeids``: node positions are
  global across trees, each tree is rooted via ``tree_roots``, and a
  branch references either an interior-node position (when the matching
  ``*_leafs`` flag is 0) or a leaf-array position (flag is 1).
- ``membership_values`` concatenates, for each ``BRANCH_MEMBER`` node in
  ``nodes_modes`` order, its member values terminated by a single NaN.
- Each leaf-array tuple contributes one ``(leaf_targetids, leaf_weights)``
  pair to one output target, so classification duplicates every tree once
  per class with a constant per-copy target id; summed scores then equal
  the averaged class probabilities (mirroring the generic exporter).
- A degenerate single-leaf tree is represented as one dummy interior node
  whose both branches reference its only leaf slot.
"""

from __future__ import annotations

import numpy as np

from shinrin.categorical import _encoder_tables
from shinrin.onnx import _require_onnx, _set_props, _tree_iter

# ai.onnx.ml domain version introducing TreeEnsemble with BRANCH_MEMBER.
AI_ONNX_ML_MEMBER_OPSET = 5

_BRANCH_LEQ = 0
_BRANCH_MEMBER = 6


def _tensor_attr(name: str, values, dtype_code: int):
    """Build a tensor-typed attribute (v5 requires tensors, not FLOATS)."""
    from onnx import helper

    arr = np.ascontiguousarray(np.asarray(values))
    return helper.make_attribute(
        name,
        helper.make_tensor(name, dtype_code, list(arr.shape), arr.ravel()),
    )


class _MemberBuilder:
    """Accumulates opset-5 ``TreeEnsemble`` attribute arrays.

    Each call to :meth:`add_tree` appends one *emitted* tree (for
    classification the caller emits one copy per class). Leaves map 1:1 to
    leaf-array slots; :attr:`leaves_by_tree` records each leaf's slot and
    raw ``tree_.value`` vector so the caller can fill ``leaf_weights`` /
    ``leaf_targetids``.
    """

    def __init__(self) -> None:
        self.nodes_featureids: list[int] = []
        self.nodes_modes: list[int] = []
        self.nodes_splits: list[float] = []
        self.nodes_truenodeids: list[int] = []
        self.nodes_trueleafs: list[int] = []
        self.nodes_falsenodeids: list[int] = []
        self.nodes_falseleafs: list[int] = []
        self.membership_values: list[float] = []
        self.tree_roots: list[int] = []
        # Per emitted tree: {node_index: (leaf_slot, value_vector)}.
        self.leaves_by_tree: list[dict[int, tuple[int, np.ndarray]]] = []
        # v5 leaf-array positions are global across trees.
        self._leaf_offset = 0

    def add_tree(
        self,
        tree,
        cat_features: set[int],
        cats: dict[int, np.ndarray],
        encs: dict[int, np.ndarray],
    ) -> None:
        """Append one hard tree structure (sklearn or native Mondrian)."""
        left = np.asarray(tree.children_left)
        right = np.asarray(tree.children_right)
        feature = np.asarray(tree.feature)
        threshold = np.asarray(tree.threshold, dtype=np.float64)
        value = np.asarray(tree.value, dtype=np.float64)

        n_nodes = len(left)

        if n_nodes == 1:
            # Degenerate single-leaf tree: represent it as one dummy
            # interior node whose both branches hit the same leaf slot.
            self.tree_roots.append(len(self.nodes_modes))
            self.nodes_featureids.append(0)
            self.nodes_modes.append(_BRANCH_LEQ)
            self.nodes_splits.append(0.0)
            self.nodes_truenodeids.append(0)
            self.nodes_trueleafs.append(1)
            self.nodes_falsenodeids.append(0)
            self.nodes_falseleafs.append(1)
            self.leaves_by_tree.append({0: (self._leaf_offset, value[0].reshape(-1))})
            self._leaf_offset += 1
            return

        node_pos: dict[int, int] = {}
        leaves: dict[int, tuple[int, np.ndarray]] = {}
        next_leaf_slot = self._leaf_offset
        # Internal sklearn nodes always have both children.
        parent = np.full(n_nodes, -1, dtype=np.int64)
        for j in range(n_nodes):
            if left[j] >= 0:
                parent[left[j]] = j
                parent[right[j]] = j

        for i in range(n_nodes):
            is_leaf = left[i] < 0 and right[i] < 0
            if is_leaf:
                leaves[i] = (next_leaf_slot, value[i].reshape(-1))
                next_leaf_slot += 1
            else:
                f = int(feature[i])
                node_pos[i] = len(self.nodes_modes)
                self.nodes_featureids.append(f)
                self.nodes_truenodeids.append(-1)
                self.nodes_trueleafs.append(0)
                self.nodes_falsenodeids.append(-1)
                self.nodes_falseleafs.append(0)
                if f in cat_features:
                    self.nodes_modes.append(_BRANCH_MEMBER)
                    self.nodes_splits.append(0.0)
                    members = np.sort(cats[f][encs[f] <= threshold[i]])
                    if len(members) == 0:
                        raise ValueError(
                            f"split on encoded feature {f} selects no "
                            "category; encoder tables disagree with the "
                            "trained model"
                        )
                    self.membership_values.extend(float(c) for c in members)
                    self.membership_values.append(np.nan)
                else:
                    self.nodes_modes.append(_BRANCH_LEQ)
                    self.nodes_splits.append(float(threshold[i]))

            # Wire this node into its parent edge (root handled below).
            p = int(parent[i])
            if p < 0:
                # Root of a non-degenerate tree is always an interior node.
                self.tree_roots.append(node_pos[i])
                continue
            ppos = node_pos[p]
            if is_leaf:
                slot = leaves[i][0]
                if left[p] == i:
                    self.nodes_truenodeids[ppos] = slot
                    self.nodes_trueleafs[ppos] = 1
                else:
                    self.nodes_falsenodeids[ppos] = slot
                    self.nodes_falseleafs[ppos] = 1
            elif left[p] == i:
                self.nodes_truenodeids[ppos] = node_pos[i]
            else:
                self.nodes_falsenodeids[ppos] = node_pos[i]

        self.leaves_by_tree.append(leaves)
        self._leaf_offset += len(leaves)


def _collect_model_data(estimator, encoder):
    """Validate encoder/model agreement; return shared export inputs."""
    feats, cats, encs = _encoder_tables(encoder)
    trees, is_classification = _tree_iter(estimator)
    n_features = int(trees[0].n_features)
    enc_n = getattr(encoder, "n_features_in_", None)
    if enc_n is not None and int(enc_n) != n_features:
        raise ValueError(
            f"encoder was fitted on {int(enc_n)} columns but the model "
            f"expects {n_features} features"
        )
    unknown = [f for f in feats if f >= n_features]
    if unknown:
        raise ValueError(
            f"encoder covers categorical features {unknown} outside the "
            f"model's {n_features}-feature input"
        )
    return trees, is_classification, n_features, set(feats), cats, encs


def treeensemble_member_to_onnx(
    estimator,
    encoder,
    feature_names=None,
    class_names=None,
    name="ShinrinModel",
    target_opset=15,
    approximate: bool | None = None,
):
    """Export a fitted tree/forest to an opset-5 ``BRANCH_MEMBER`` graph.

    Parameters
    ----------
    estimator : fitted tree or forest
        Model trained on target-encoded features (shinrin Mondrian
        tree/forest or scikit-learn-style ``tree_`` holders).
    encoder : fitted target encoder
        :class:`shinrin.TargetEncoder` (or duck-typed equivalent exposing
        ``categorical_features_``, ``categories_``, ``encodings_``) used
        to encode categorical columns before training.
    feature_names : list of str, optional
        Stored as model metadata.
    class_names : list of str, optional
        When provided (classification), the ``labels`` output yields these
        names instead of integer class values.
    name : str
        Model/producer name.
    target_opset : int
        Default-domain opset version (default 15).
    approximate : bool, optional
        Unused; accepted so callers can pass a uniform keyword set.

    Returns
    -------
    onnx.ModelProto consuming float32 raw-code ``X`` of shape ``(batch,
    n_features)``. Outputs match the generic exporter: regression produces
    ``predictions`` of shape ``(batch,)``; classification produces
    ``labels`` and ``probabilities`` ``(batch, n_classes)``.
    """
    TensorProto, helper = _require_onnx()
    del approximate  # accepted for API symmetry with mondrian_to_onnx

    (
        trees,
        is_classification,
        n_features,
        cat_features,
        cats,
        encs,
    ) = _collect_model_data(estimator, encoder)
    n_trees = len(trees)

    # GradientBoosting ensembles sum learning-rate-scaled tree outputs on
    # top of a constant; forests plain-average raw outputs.
    is_gb = type(estimator).__name__.startswith("GradientBoosting")
    if is_gb:
        if is_classification or getattr(estimator, "classes_", None) is not None:
            raise NotImplementedError(
                "member export for GradientBoosting classifiers is not "
                "supported; use GradientBoostingRegressor or a forest model."
            )
        learning_rate = float(getattr(estimator, "learning_rate", 1.0))
        base_value = float(np.ravel(estimator.init_.constant_)[0])
    else:
        learning_rate = 1.0
        base_value = 0.0

    n_classes = (
        max(int(np.asarray(t.n_classes).ravel()[0]) for t in trees)
        if is_classification
        else 1
    )

    builder = _MemberBuilder()
    # Classification duplicates each tree once per class so that every
    # leaf slot carries a single (target=k, weight=P(k|leaf)) pair.
    copies_per_tree = n_classes if is_classification else 1
    for _k in range(copies_per_tree):
        for tree in trees:
            builder.add_tree(tree, cat_features, cats, encs)

    leaf_targetids: list[int] = []
    leaf_weights: list[float] = []
    scale = 1.0 / n_trees
    # Forests average their trees' outputs; folding 1/T into the weights
    # keeps SUM aggregation correct. GB trees keep their raw lr-scaled sums.
    weight_factor = learning_rate if is_gb else scale
    for k in range(copies_per_tree):
        for leaves in builder.leaves_by_tree[k * n_trees : (k + 1) * n_trees]:
            for _node, (_slot, vec) in sorted(leaves.items()):
                leaf_targetids.append(k)
                if is_classification:
                    total = float(vec.sum())
                    probs = (
                        vec / total if total > 0 else np.full_like(vec, 1.0 / n_classes)
                    )
                    leaf_weights.append(float(probs[k]) * scale)
                else:
                    leaf_weights.append(float(vec[0]) * weight_factor)

    attrs = {
        "n_targets": n_classes if is_classification else 1,
        "aggregate_function": 1,  # SUM; averaging folded into the weights
        "post_transform": 0,  # NONE
        "tree_roots": builder.tree_roots,
        "nodes_featureids": builder.nodes_featureids,
        "nodes_truenodeids": builder.nodes_truenodeids,
        "nodes_trueleafs": builder.nodes_trueleafs,
        "nodes_falsenodeids": builder.nodes_falsenodeids,
        "nodes_falseleafs": builder.nodes_falseleafs,
        "leaf_targetids": leaf_targetids,
    }
    out_name = "probabilities" if is_classification else "te_out"
    node = helper.make_node(
        "TreeEnsemble",
        inputs=["X"],
        outputs=[out_name],
        domain="ai.onnx.ml",
        **attrs,
    )
    node.attribute.append(
        _tensor_attr("nodes_modes", builder.nodes_modes, TensorProto.UINT8)
    )
    node.attribute.append(
        _tensor_attr("nodes_splits", builder.nodes_splits, TensorProto.FLOAT)
    )
    node.attribute.append(
        _tensor_attr(
            "membership_values",
            np.array(builder.membership_values, dtype=np.float32),
            TensorProto.FLOAT,
        )
    )
    node.attribute.append(_tensor_attr("leaf_weights", leaf_weights, TensorProto.FLOAT))

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    opset_imports = [
        helper.make_opsetid("", target_opset),
        helper.make_opsetid("ai.onnx.ml", AI_ONNX_ML_MEMBER_OPSET),
    ]

    if not is_classification:
        initializers = [helper.make_tensor("neg1", TensorProto.INT64, [1], [-1])]
        nodes = [node]
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
            nodes, f"{name}_graph", inputs, tail_outputs, initializer=initializers
        )
    else:
        classes = np.asarray(getattr(estimator, "classes_", np.arange(n_classes)))
        if classes.shape[0] != n_classes:  # pragma: no cover
            classes = np.arange(n_classes)
        if np.issubdtype(classes.dtype, np.number):
            cls_init = helper.make_tensor(
                "classes",
                TensorProto.INT64,
                list(classes.shape),
                classes.astype(np.int64).ravel(),
            )
            labels_dtype = TensorProto.INT64
        else:
            cls_init = helper.make_tensor(
                "classes",
                TensorProto.STRING,
                list(classes.shape),
                np.array([str(c) for c in classes], object).ravel(),
            )
            labels_dtype = TensorProto.STRING
        nodes = [
            node,
            helper.make_node("ArgMax", ["probabilities"], ["amax"], axis=1, keepdims=0),
            helper.make_node("Unsqueeze", ["amax", "cls_ax"], ["amax4g"]),
            helper.make_node("Gather", ["classes", "amax4g"], ["labels2d"], axis=0),
            helper.make_node("Squeeze", ["labels2d", "cls_ax"], ["labels"]),
        ]
        initializers = [
            cls_init,
            helper.make_tensor("cls_ax", TensorProto.INT64, [1], [1]),
        ]
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

    props = {
        "shinrin_version": "0.2.0",
        "model_type": type(estimator).__name__,
        "shinrin_treeensemble_export": "member-v5",
    }
    if feature_names is not None:
        props["feature_names"] = ",".join(str(f) for f in feature_names)
    if class_names is not None and is_classification:
        props["class_names"] = ",".join(str(c) for c in class_names)
    _set_props(model, props)
    return model
