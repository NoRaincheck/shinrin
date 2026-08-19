"""ONNX exporter for shinrin tree and forest models.

This module provides functionality to export fitted shinrin tree and forest
models to the ONNX format, enabling deployment on platforms that support
ONNX runtime inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy import for ONNX types (used in _onnx_dtype)
try:
    from onnx import TensorProto  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover
    TensorProto = None  # type: ignore[misc,assignment]


def _tree_to_onnx(
    tree,
    feature_names=None,
    label_names=None,
    class_names=None,
    default_tree_id=0,
):
    """Convert a single tree to ONNX tree ensemble node attributes.

    Parameters
    ----------
    tree : fitted tree estimator with ``tree_`` attribute
        The tree to export.
    feature_names : list of str, optional
        Names of input features.
    label_names : list of str, optional
        Names of output labels.
    class_names : list of str, optional
        Class names for classification models.
    default_tree_id : int
        Default tree ID for the ONNX node.

    Returns
    -------
    dict
        ONNX node attributes for this tree.
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(tree.tree_.n_features)]

    t = tree.tree_
    n_nodes = t.node_count
    is_regression = int(t.n_classes[0]) == 1

    # Build node arrays
    node_ids = np.arange(n_nodes, dtype=np.int64)
    feature_ids = t.feature.astype(np.int64)
    left_child_ids = t.children_left.astype(np.int64)
    right_child_ids = t.children_right.astype(np.int64)
    node_values = t.value

    # ONNX TreeEnsemble node expects specific attribute formats
    attributes = {
        "name": f"tree_{default_tree_id}",
        "tree_roots": np.array([0], dtype=np.int64),
    }

    if is_regression:
        # For regression, values are the predicted mean at each node
        # Reshape to (n_nodes, 1) for single output
        if node_values.ndim == 3:
            # (n_nodes, n_outputs, max_n_classes) -> (n_nodes,)
            reg_values = node_values[:, 0, 0]
        else:
            reg_values = node_values.ravel()

        attributes.update(
            {
                "base_values": np.array([0.0], dtype=np.float64),
                "nodes_feature_ids": feature_ids,
                "nodes_missing_value_tracks_true": np.zeros(n_nodes, dtype=np.int64),
                "nodes_hitrates": np.zeros(n_nodes, dtype=np.float32),
                "nodes_node_ids": node_ids,
                "nodes_tree_id": np.full(n_nodes, default_tree_id, dtype=np.int64),
                "nodes_truenode_ids": np.where(
                    left_child_ids == -1, -1, left_child_ids
                ),
                "nodes_falsenode_ids": np.where(
                    right_child_ids == -1, -1, right_child_ids
                ),
                "nodes_values": reg_values.astype(np.float64),
                "post_transform": "none",
            }
        )
    else:
        # For classification, we need per-class values
        n_classes = int(t.n_classes[0])
        if n_classes == 2:
            # Binary classification: one value array, transform is sigmoid
            if node_values.ndim == 3:
                clf_values = node_values[:, 0, 1] - node_values[:, 0, 0]
            else:
                clf_values = node_values.ravel()

            attributes.update(
                {
                    "base_values": np.array([0.0], dtype=np.float64),
                    "nodes_feature_ids": feature_ids,
                    "nodes_missing_value_tracks_true": np.zeros(
                        n_nodes, dtype=np.int64
                    ),
                    "nodes_hitrates": np.zeros(n_nodes, dtype=np.float32),
                    "nodes_node_ids": node_ids,
                    "nodes_tree_id": np.full(n_nodes, default_tree_id, dtype=np.int64),
                    "nodes_truenode_ids": np.where(
                        left_child_ids == -1, -1, left_child_ids
                    ),
                    "nodes_falsenode_ids": np.where(
                        right_child_ids == -1, -1, right_child_ids
                    ),
                    "nodes_values": clf_values.astype(np.float64),
                    "post_transform": "sigmoid",
                }
            )
        else:
            # Multi-class: one value array per class
            if node_values.ndim == 3:
                clf_values = node_values[:, 0, :n_classes].flatten()
            else:
                clf_values = node_values.ravel()

            attributes.update(
                {
                    "base_values": np.zeros(n_classes, dtype=np.float64),
                    "nodes_feature_ids": feature_ids,
                    "nodes_missing_value_tracks_true": np.zeros(
                        n_nodes, dtype=np.int64
                    ),
                    "nodes_hitrates": np.zeros(n_nodes, dtype=np.float32),
                    "nodes_node_ids": node_ids,
                    "nodes_tree_id": np.full(n_nodes, default_tree_id, dtype=np.int64),
                    "nodes_truenode_ids": np.where(
                        left_child_ids == -1, -1, left_child_ids
                    ),
                    "nodes_falsenode_ids": np.where(
                        right_child_ids == -1, -1, right_child_ids
                    ),
                    "nodes_values": clf_values.astype(np.float64),
                    "post_transform": "softmax",
                }
            )

    return attributes


def to_onnx(
    estimator,
    X=None,
    feature_names=None,
    class_names=None,
    name="ShinrinTree",
    target_opset=None,
):
    """Convert a fitted shinrin tree or forest model to ONNX format.

    Parameters
    ----------
    estimator : fitted tree or forest estimator
        The model to export. Must have ``tree_`` (single tree) or
        ``estimators_`` (forest) attribute.
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
        ONNX opset version. Defaults to 15.

    Returns
    -------
    onnx.ModelProto
        The ONNX model representation.

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
        from onnx import (  # ty: ignore[unresolved-import]
            TensorProto,
            helper,
            numpy_helper,
        )
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )

    if target_opset is None:
        target_opset = 15

    # Infer input shape and dtype
    if X is not None:
        X = np.asarray(X)
        n_features = X.shape[1]
        input_dtype = X.dtype
    else:
        n_features = 4
        input_dtype = np.float64

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # Determine if regression or classification
    if hasattr(estimator, "tree_"):
        is_classification = int(estimator.tree_.n_classes[0]) > 1
        n_classes = (
            int(estimator.tree_.n_classes[0])
            if len(estimator.tree_.n_classes) > 0
            else 1
        )
    elif hasattr(estimator, "estimators_"):
        # Forest: use first estimator to determine type
        first_tree = estimator.estimators_[0]
        if hasattr(first_tree, "tree_"):
            is_classification = int(first_tree.tree_.n_classes[0]) > 1
            n_classes = (
                int(first_tree.tree_.n_classes[0])
                if len(first_tree.tree_.n_classes) > 0
                else 1
            )
        else:
            raise ValueError("Forest estimators must contain tree models.")
    else:
        raise ValueError(
            "Estimator must have a 'tree_' attribute (single tree) or "
            "'estimators_' attribute (forest)."
        )

    # Create input tensor
    input_name = feature_names[0] if len(feature_names) == 1 else "X"
    input_tensor = helper.make_tensor_value_info(
        input_name,
        _onnx_dtype(input_dtype),
        [None, n_features],
    )

    # Build tree ensemble nodes
    if hasattr(estimator, "tree_"):
        # Single tree
        tree_attrs = _tree_to_onnx(
            estimator,
            feature_names=feature_names,
            class_names=class_names,
        )
        tree_nodes = [tree_attrs]
    else:
        # Forest: one tree ensemble node per estimator
        tree_nodes = []
        for i, tree in enumerate(estimator.estimators_):
            tree_attrs = _tree_to_onnx(
                tree,
                feature_names=feature_names,
                class_names=class_names,
                default_tree_id=i,
            )
            tree_nodes.append(tree_attrs)

    # Create output
    if is_classification:
        if class_names is not None and n_classes > 2:
            label_tensor = helper.make_tensor_value_info(
                "labels",
                TensorProto.STRING,
                [None],
            )
        else:
            label_tensor = helper.make_tensor_value_info(
                "labels",
                TensorProto.INT64,
                [None],
            )
        output_name = "probabilities" if n_classes == 2 else "class_labels"
        if n_classes == 2:
            output_tensor = helper.make_tensor_value_info(
                output_name,
                TensorProto.DOUBLE,
                [None, 2],
            )
        else:
            output_tensor = helper.make_tensor_value_info(
                output_name,
                TensorProto.INT64,
                [None],
            )
    else:
        output_name = "predictions"
        output_tensor = helper.make_tensor_value_info(
            output_name,
            TensorProto.DOUBLE,
            [None],
        )
        label_tensor = None

    # Build ONNX model
    if len(tree_nodes) == 1:
        # Single tree: simple TreeEnsemble node
        tree_node = helper.make_node(
            "TreeEnsemble",
            inputs=[input_name],
            outputs=["tree_output"],
            **tree_nodes[0],
        )
        if is_classification and n_classes == 2:
            # Sigmoid output already handled in TreeEnsemble
            output_node = helper.make_node(
                "Identity",
                inputs=["tree_output"],
                outputs=[output_name],
            )
            graph = helper.make_graph(
                [tree_node, output_node],
                f"{name}_graph",
                [input_tensor],
                [output_tensor],
            )
        elif is_classification and n_classes > 2:
            # Softmax output already handled in TreeEnsemble
            # Add label mapping
            if class_names is not None:
                class_map = numpy_helper.from_array(
                    np.array(class_names, dtype=object),
                    name="class_names",
                )
                graph = helper.make_graph(
                    [tree_node],
                    f"{name}_graph",
                    [input_tensor],
                    [output_tensor, label_tensor],
                )
                model = helper.make_model(
                    graph,
                    opset_imports=[
                        helper.make_opsetid("", target_opset),
                    ],
                    producer_name=name,
                    ir_version=8,
                )
                model.graph.initializer.append(class_map)
            else:
                graph = helper.make_graph(
                    [tree_node],
                    f"{name}_graph",
                    [input_tensor],
                    [output_tensor],
                )
                model = helper.make_model(
                    graph,
                    opset_imports=[
                        helper.make_opsetid("", target_opset),
                    ],
                    producer_name=name,
                    ir_version=8,
                )
        else:
            graph = helper.make_graph(
                [tree_node],
                f"{name}_graph",
                [input_tensor],
                [output_tensor],
            )
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", target_opset),
                ],
                producer_name=name,
                ir_version=8,
            )
    else:
        # Forest: multiple TreeEnsemble nodes + ReduceSum
        tree_outputs = [f"tree_out_{i}" for i in range(len(tree_nodes))]
        tree_nodes_onnx = []
        for i, attrs in enumerate(tree_nodes):
            node = helper.make_node(
                "TreeEnsemble",
                inputs=[input_name],
                outputs=[tree_outputs[i]],
                **attrs,
            )
            tree_nodes_onnx.append(node)

        # Average predictions across trees
        if is_classification and n_classes == 2:
            # For binary classification, average sigmoid outputs
            if len(tree_outputs) > 1:
                reduce_node = helper.make_node(
                    "ReduceMean",
                    inputs=tree_outputs,
                    outputs=["forest_output"],
                    keepdims=1,
                )
                # Reshape to (n_samples, 2)
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=["forest_output", "shape"],
                    outputs=[output_name],
                )
                shape_tensor = numpy_helper.from_array(
                    np.array([-1, 2], dtype=np.int64),
                    name="shape",
                )
                graph = helper.make_graph(
                    tree_nodes_onnx + [reduce_node, reshape_node],
                    f"{name}_graph",
                    [input_tensor],
                    [output_tensor],
                )
                graph.initializer.append(shape_tensor)
                model = helper.make_model(
                    graph,
                    opset_imports=[
                        helper.make_opsetid("", target_opset),
                    ],
                    producer_name=name,
                    ir_version=8,
                )
                return model
        elif is_classification and n_classes > 2:
            # For multi-class, average probabilities
            reduce_node = helper.make_node(
                "ReduceMean",
                inputs=tree_outputs,
                outputs=["forest_output"],
                keepdims=0,
            )
            softmax_node = helper.make_node(
                "Softmax",
                inputs=["forest_output"],
                outputs=[output_name],
                axis=1,
            )
            graph = helper.make_graph(
                tree_nodes_onnx + [reduce_node, softmax_node],
                f"{name}_graph",
                [input_tensor],
                [output_tensor],
            )
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", target_opset),
                ],
                producer_name=name,
                ir_version=8,
            )
            return model
        else:
            # Regression: average tree outputs
            reduce_node = helper.make_node(
                "ReduceMean",
                inputs=tree_outputs,
                outputs=[output_name],
                keepdims=0,
            )
            graph = helper.make_graph(
                tree_nodes_onnx + [reduce_node],
                f"{name}_graph",
                [input_tensor],
                [output_tensor],
            )
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", target_opset),
                ],
                producer_name=name,
                ir_version=8,
            )
            return model

    # Add metadata
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", target_opset),
        ],
        producer_name=name,
        ir_version=8,
    )
    model.metadata_props.append(
        helper.make_string_key_value_pair("shinrin_version", "0.1.0"),
    )
    return model


def _onnx_dtype(dtype):
    """Map numpy dtype to ONNX TensorProto dtype."""
    try:
        from onnx import TensorProto  # ty: ignore[unresolved-import]
    except ImportError:
        TensorProto = None  # type: ignore[misc,assignment]

    mapping: dict[Any, Any] = {
        np.float16: "FLOAT",
        np.float32: "FLOAT",
        np.float64: "DOUBLE",
        np.int32: "INT32",
        np.int64: "INT64",
        np.int8: "INT8",
        np.uint8: "UINT8",
        np.bool_: "BOOL",
    }
    type_name = mapping.get(dtype, "FLOAT")
    if TensorProto is not None:
        return getattr(TensorProto, type_name)
    return type_name


def save_onnx(estimator, path, X=None, feature_names=None, class_names=None):
    """Save a fitted shinrin model to an ONNX file.

    Parameters
    ----------
    estimator : fitted tree or forest estimator
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
