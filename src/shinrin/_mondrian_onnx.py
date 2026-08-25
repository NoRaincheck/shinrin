"""ONNX export for Mondrian trees and forests.

Two encodings are produced, selected by the estimator's
``path_smoothing`` prediction mode (see ``mondrian_to_onnx``):

- ``tree-ensemble`` (default models): a plain ``ai.onnx.ml``
  tree-ensemble of the hard tree structure. Native constant-mode
  prediction routes each sample down a single root-to-leaf path by hard
  threshold comparisons and returns the leaf value, which is exactly
  what this graph computes.
- ``exact`` (`path_smoothing=True` models): a standard-domain graph
  reproducing the weighted-path prediction of ``shinrin-native``
  ``predict``, where every visited node contributes with a
  Mondrian-process weight:

  - ``eta_j(x)  = sum_f max(x_f - upper_jf, 0) + max(lower_jf - x_f, 0)``
    (distance of ``x`` outside node ``j``'s bounding box),
  - ``delta_j   = tau_j - tau_parent(j)``,
  - ``p_js_j(x) = 1 - exp(-delta_j * eta_j(x))``,
  - survival along the path: ``p_nsy *= 1 - p_js_j`` after each internal node,
  - per-node weight: ``w_j = p_nsy_before(j) * p_js_j`` for internal nodes,
    ``w_leaf = p_nsy_at_leaf`` (no eta factor),
  - routing: ``x_f <= threshold_j`` goes left, else right,

  and the prediction is ``sum_j w_j * value_j`` over the visited nodes
  (regression uses raw values; classification normalises by node sample
  counts first).

Graph layout
------------
Nodes are ordered breadth-first. For all internal nodes we evaluate
``q = exp(-delta * eta)`` and the hard branch bits ``il = [thr >= x_feat]``
/ ``ir = 1 - il``, each of shape ``(batch, k)``. Fixed 0/1 selection
matrices scatter them to full BFS width ``(batch, n_total)``:

- ``QFULL[:, c]  = q_parent(c)``   (0 at the root)
- ``RFULL[:, c]  = bit into child c from its parent``  (1 at the root)
- ``PFULL[:, c]  = p_js_c`` for internal nodes, 1 for leaves

Per depth level we track the level's internal-node survival factors ``SP``
(parent width), duplicate them per child pair, and combine with sliced
windows of the full-width matrices:

- ``S_d = SDUP_{d-1} * QWIN_d * RWIN_d``  (survival into each level-d node,
  zero for off-path nodes via the routing bits)
- ``W_d = S_d * PWIN_d``                  (per-node weight block)

Forests concatenate per-tree ``W`` blocks; a single ``MatMul(W, values)``
followed by division by the tree count yields averaged predictions.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover
    TensorProto: Any = None
    helper: Any = None


# ---------------------------------------------------------------------------
# Tree -> BFS-layout arrays
# ---------------------------------------------------------------------------


class _FlatTree:
    """A Mondrian tree flattened into BFS node order."""

    def __init__(self, tree) -> None:
        left = np.asarray(tree.children_left)
        right = np.asarray(tree.children_right)
        feat = np.asarray(tree.feature)
        thr = np.asarray(tree.threshold, dtype=np.float64)
        tau = np.asarray(tree.tau, dtype=np.float64)
        upper = np.asarray(tree.upper_bounds, dtype=np.float64)
        lower = np.asarray(tree.lower_bounds, dtype=np.float64)
        value = np.asarray(tree.value, dtype=np.float64)
        n_node_samples = np.asarray(tree.n_node_samples, dtype=np.float64)

        root = int(getattr(tree, "root", 0))

        # BFS traversal recording each node's parent tau along the way.
        levels: list[list[tuple[int, float]]] = []
        current = [(root, 0.0)]
        while current:
            levels.append(current)
            nxt: list[tuple[int, float]] = []
            for nid, ptau in current:
                if left[nid] >= 0:
                    nxt.append((int(left[nid]), float(tau[nid])))
                if right[nid] >= 0:
                    nxt.append((int(right[nid]), float(tau[nid])))
            current = nxt

        order = [nid for level in levels for nid, _ in level]
        ptau_map: dict[int, float] = {
            nid: ptau for level in levels for nid, ptau in level
        }
        bfs_pos = {nid: p for p, nid in enumerate(order)}

        self.level_sizes = [len(level) for level in levels]
        self.level_offsets = np.concatenate([[0], np.cumsum(self.level_sizes)]).astype(
            np.int64
        )
        # Positions (within their level) of internal nodes at each level;
        # used to compact full-width level vectors down to parent width.
        self.internal_positions = [
            [p for p, (nid, _) in enumerate(level) if left[nid] >= 0]
            for level in levels
        ]

        # Internal nodes in global BFS order.
        internal_ids = [nid for nid in order if left[nid] >= 0]
        k = len(internal_ids)
        self.n_internal = k
        self.n_total = len(order)

        # BFS columns holding q / routing bits / p_js.
        self.internal_cols = np.array(
            [bfs_pos[nid] for nid in internal_ids], dtype=np.int64
        )
        # BFS columns of each internal node's children (left/right).
        self.left_child_cols = np.array(
            [bfs_pos[int(left[nid])] for nid in internal_ids], dtype=np.int64
        )
        self.right_child_cols = np.array(
            [bfs_pos[int(right[nid])] for nid in internal_ids], dtype=np.int64
        )

        self.feat_idx = np.array(
            [int(feat[nid]) for nid in internal_ids], dtype=np.int64
        )
        # Round split thresholds DOWN to the next float32 so training points
        # sitting exactly on a split keep going left (x <= t), matching f64.
        self.thresholds = np.array(
            [
                np.nextafter(np.float32(thr[nid]), np.float32(-np.inf))
                if float(np.float32(thr[nid])) > float(thr[nid])
                else np.float32(thr[nid])
                for nid in internal_ids
            ],
            dtype=np.float32,
        )
        self.delta_neg = np.array(
            [-(float(tau[nid]) - float(ptau_map[nid])) for nid in internal_ids],
            dtype=np.float64,
        )
        if k:
            self.upper = upper[np.array(internal_ids)].astype(np.float32)
            self.lower = lower[np.array(internal_ids)].astype(np.float32)
        else:
            self.upper = np.zeros((0, int(tree.n_features)), np.float32)
            self.lower = np.zeros((0, int(tree.n_features)), np.float32)

        # Reorder node attributes into BFS column order so they align with
        # the weight vectors produced by the level recursion.
        self.values = value[np.array(order)].reshape(len(order), -1).astype(np.float64)
        self.n_node_samples = n_node_samples[np.array(order)]
        self.n_features = int(tree.n_features)


def _collect_trees(estimator) -> list[_FlatTree]:
    if hasattr(estimator, "tree_"):
        return [_FlatTree(estimator.tree_)]
    estimators = getattr(estimator, "estimators_", None)
    if estimators is None:
        raise ValueError(
            f"Estimator of type {type(estimator).__name__} is not a fitted "
            "tree or forest: expected 'tree_' or 'estimators_'."
        )
    if getattr(estimators, "ndim", 1) > 1:
        estimators = [e[0] for e in estimators]
    return [_FlatTree(e.tree_) for e in estimators]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


# Exact Mondrian graphs carry per-tree ``(internal_nodes x total_nodes)``
# 0/1 selection matrices; their combined float32 size grows roughly with
# tree count x nodes squared. Past this many estimated initializer bytes
# the proto risks the 2 GB protobuf hard limit and slow runtimes, so
# ``mondrian_to_onnx`` falls back to a plain ``ai.onnx.ml`` tree-ensemble
# export of the hard tree structure (see the ``approximate`` parameter).
MONDRIAN_EXACT_MAX_BYTES = 256 * 1024 * 1024

PROP_EXPORT_MODE = "shinrin_mondrian_export"
MODE_EXACT = "exact"
MODE_TREE_ENSEMBLE = "tree-ensemble"


def _estimated_exact_bytes(trees) -> int:
    """Estimate initializer bytes of the exact graph for these trees.

    The builder concatenates every tree into shared tables, so the four
    ``(internal_nodes x total_nodes)`` selection matrices scale with the
    *product* of the totals, not the sum of per-tree products.
    """
    k_total = sum(int(t.delta_neg.size) for t in trees)  # internal nodes
    n_total = sum(int(t.n_total) for t in trees)
    n_classes = max(1, max(t.values.shape[1] for t in trees))
    n_features = trees[0].n_features
    # gt_q/gl/gr/gp selection matrices dominate.
    return int(
        4 * (4 * k_total * n_total + n_total * n_classes + k_total * (n_features + 4))
    )


def mondrian_to_onnx(
    estimator,
    feature_names=None,
    class_names=None,
    name="ShinrinMondrianModel",
    target_opset=15,
    approximate: bool | None = None,
    encoder=None,
):
    """Export a fitted Mondrian tree/forest to ONNX.

    The encoding follows the estimator's ``path_smoothing`` prediction
    mode so that exported predictions match native ``predict`` /
    ``predict_proba`` exactly:

    - Constant prediction (``path_smoothing=False``, the default): a plain
      ``ai.onnx.ml`` tree-ensemble of the hard tree structure (leaf means /
      class distributions averaged across trees). This *is* the native
      model, so the result is exact, small and fast.
    - Path smoothing (``path_smoothing=True``): by default an exact
      standard-domain graph reproducing the Mondrian-process smoothing
      along decision paths to float32 round-off. Because exact graphs
      embed per-node selection matrices whose size grows with node count
      squared, very large ensembles automatically fall back to the plain
      tree-ensemble encoding; that fallback omits the smoothing, so its
      predictions then deviate from native ``predict``.

    Parameters
    ----------
    estimator : fitted MondrianTree* / MondrianForest*
        The model to export.
    feature_names : list of str, optional
        Stored as model metadata.
    class_names : list of str, optional
        When provided (classification), the ``labels`` output yields these
        names instead of integer class values. Only honored by the exact
        export.
    name : str
        Model/producer name.
    target_opset : int
        Default-domain opset version (default 15).
    approximate : bool, optional
        Force the plain tree-ensemble encoding (``True``) or the exact
        smoothing graph (``False``). By default (``None``) the encoding is
        chosen from the estimator's ``path_smoothing`` mode; smoothing
        models additionally fall back to the tree-ensemble when the exact
        graph's estimated initializer size exceeds
        ``MONDRIAN_EXACT_MAX_BYTES`` (a :class:`UserWarning` is emitted).
    encoder : fitted TargetEncoder, optional
        Target encoder used to encode categorical columns before training.
        When provided, forces the hard tree-structure encoding and emits an
        ``ai.onnx.ml`` opset-5 ``TreeEnsemble`` with ``BRANCH_MEMBER``
        categorical splits consuming raw category codes. Path smoothing
        cannot be represented, so smoothing estimators export their hard
        structure (a :class:`UserWarning` is emitted); pass
        ``approximate=False`` to get the exact smoothing graph instead.

    Returns
    -------
    onnx.ModelProto consuming float32 ``X`` of shape ``(batch,
    n_features)``. Outputs match the generic exporter: regression produces
    ``predictions`` of shape ``(batch,)``; classification produces
    ``labels`` (int64 class values) and ``probabilities`` ``(batch,
    n_classes)``. The metadata property ``shinrin_mondrian_export`` records
    which encoding was produced (``"exact"`` or ``"tree-ensemble"``).
    """
    if helper is None or TensorProto is None:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )
    trees = _collect_trees(estimator)
    smoothing = bool(getattr(estimator, "path_smoothing", True))

    if encoder is not None and approximate is not False:
        # Member export always encodes the hard tree structure; warn when
        # native predict() would smooth along paths because that behaviour
        # cannot survive a tree-ensemble encoding.
        if smoothing:
            warnings.warn(
                "encoder= exports the hard tree structure with BRANCH_MEMBER "
                "categorical splits; Mondrian path smoothing cannot be "
                "represented and predictions will deviate from native "
                "predict(). Pass approximate=False for the exact smoothing "
                "graph instead.",
                UserWarning,
                stacklevel=2,
            )
        from shinrin._treeensemble5_onnx import treeensemble_member_to_onnx

        model = treeensemble_member_to_onnx(
            estimator,
            encoder,
            feature_names=feature_names,
            class_names=class_names,
            name=name,
            target_opset=target_opset,
        )
        # set_model_props replaces all entries; merge with the exporter's.
        props = {p.key: p.value for p in model.metadata_props}
        props[PROP_EXPORT_MODE] = MODE_TREE_ENSEMBLE
        helper.set_model_props(model, props)
        return model

    use_approx = approximate
    if not smoothing:
        # Constant leaf predictions are exactly what a plain tree-ensemble
        # computes; there is no smoother to reproduce.
        if approximate is False:
            raise ValueError(
                "approximate=False requests the exact Mondrian smoothing "
                "graph, but this estimator predicts constant leaf values "
                "(path_smoothing=False). Its plain tree-ensemble export "
                "already matches native predict() exactly."
            )
        use_approx = True
    elif use_approx is None:
        use_approx = _estimated_exact_bytes(trees) > MONDRIAN_EXACT_MAX_BYTES
    if use_approx:
        if smoothing and not approximate:
            warnings.warn(
                "Exact Mondrian graph estimated at "
                f"{_estimated_exact_bytes(trees):,} initializer bytes "
                f"(limit {MONDRIAN_EXACT_MAX_BYTES:,}); falling back to a "
                "plain ai.onnx.ml tree-ensemble export of the hard tree "
                "structure. Its predictions omit Mondrian path smoothing "
                "and will deviate from native predict(). Pass "
                "approximate=False to build the exact graph anyway.",
                UserWarning,
                stacklevel=2,
            )
        from shinrin.onnx import _generic_to_onnx

        model = _generic_to_onnx(estimator, feature_names, name, target_opset)
        helper.set_model_props(model, {PROP_EXPORT_MODE: MODE_TREE_ENSEMBLE})
        return model

    model = _build_exact_model(
        estimator,
        trees,
        feature_names=feature_names,
        class_names=class_names,
        name=name,
        target_opset=target_opset,
    )
    helper.set_model_props(model, {PROP_EXPORT_MODE: MODE_EXACT})
    return model


def _build_exact_model(
    estimator,
    trees,
    feature_names=None,
    class_names=None,
    name="ShinrinMondrianModel",
    target_opset=15,
):
    """Build the exact standard-domain graph for already-collected trees."""
    is_classification = trees[0].values.shape[1] > 1
    n_features = trees[0].n_features
    n_trees = len(trees)

    inits: list = []
    nodes: list = []

    def add_init(nm, arr):
        arr = np.ascontiguousarray(arr)
        inits.append(
            helper.make_tensor(
                nm,
                TensorProto.FLOAT,
                list(arr.shape),
                arr.astype(np.float32).ravel(),
            )
        )

    def add_init_i64(nm, arr):
        arr = np.ascontiguousarray(arr)
        inits.append(
            helper.make_tensor(
                nm,
                TensorProto.INT64,
                list(arr.shape),
                arr.astype(np.int64).ravel(),
            )
        )

    add_init("one", [1.0])
    add_init_i64("ax1", [1])
    add_init_i64("ax0", [0])
    add_init_i64("ax2", [2])
    add_init_i64("idx0", [0])
    add_init_i64("one_i64", [1])
    # Dynamic (batch, 1) shape tensor.
    nodes.append(helper.make_node("Shape", ["X"], ["xshape"]))
    nodes.append(helper.make_node("Gather", ["xshape", "idx0"], ["bdim"], axis=0))
    nodes.append(helper.make_node("Concat", ["bdim", "one_i64"], ["bshape"], axis=0))

    # --- value tables in BFS row order -----------------------------------------
    if is_classification:
        n_classes = max(t.values.shape[1] for t in trees)
        rows = []
        for t in trees:
            denom = np.where(t.n_node_samples > 0, t.n_node_samples, 1.0)[:, None]
            rows.append(t.values / denom)
        add_init("vals_mat", np.concatenate(rows, axis=0))
    else:
        n_classes = 1
        add_init("vals_vec", np.concatenate([t.values[:, :1] for t in trees], axis=0))
    add_init("inv_ntrees", [1.0 / n_trees])

    # --- internal-node tables ------------------------------------------------------
    feat_idx = np.concatenate([t.feat_idx for t in trees])
    thresholds = np.concatenate([t.thresholds for t in trees])
    delta_neg = np.concatenate([t.delta_neg for t in trees])
    upper = np.concatenate([t.upper for t in trees], axis=0)
    lower = np.concatenate([t.lower for t in trees], axis=0)
    k = int(delta_neg.size)
    n_total = sum(t.n_total for t in trees)
    base_rows = [int(b) for b in np.cumsum([0] + [t.n_total for t in trees[:-1]])]

    if k:
        # eta[:, j] = sum_f relu(x_f - U_jf) + relu(L_jf - x_f), accumulated
        # feature-by-feature. Bounds stored transposed (F, k); row slices have
        # shape (1, k) so they broadcast correctly against (batch, 1).
        add_init("ut", upper.T)
        add_init("lt", lower.T)
        add_init("delta_neg", delta_neg)
        eta = ""
        for f in range(n_features):
            add_init_i64(f"f{f}a", [f])
            add_init_i64(f"f{f}b", [f + 1])
            nodes.extend(
                [
                    helper.make_node(
                        "Slice", ["X", f"f{f}a", f"f{f}b", "ax1"], [f"xf{f}"]
                    ),
                    helper.make_node(
                        "Slice", ["ut", f"f{f}a", f"f{f}b", "ax0"], [f"u{f}"]
                    ),
                    helper.make_node(
                        "Slice", ["lt", f"f{f}a", f"f{f}b", "ax0"], [f"l{f}"]
                    ),
                    helper.make_node("Sub", [f"xf{f}", f"u{f}"], [f"du{f}"]),
                    helper.make_node("Relu", [f"du{f}"], [f"ru{f}"]),
                    helper.make_node("Sub", [f"l{f}", f"xf{f}"], [f"dl{f}"]),
                    helper.make_node("Relu", [f"dl{f}"], [f"rl{f}"]),
                    helper.make_node("Add", [f"ru{f}", f"rl{f}"], [f"et{f}"]),
                ]
            )
            if not eta:
                eta = f"et{f}"
            else:
                nodes.append(helper.make_node("Add", [eta, f"et{f}"], [f"eta{f}"]))
                eta = f"eta{f}"
        # q = exp(-delta * eta); p_js = 1 - q  (internal nodes only).
        nodes.append(helper.make_node("Mul", [eta, "delta_neg"], ["qarg"]))
        nodes.append(helper.make_node("Exp", ["qarg"], ["qk"]))
        nodes.append(helper.make_node("Sub", ["one", "qk"], ["pk"]))

        # Hard branch bits: x <= thr goes left.
        add_init_i64("feat_idx", feat_idx)
        add_init("thresholds", thresholds)
        nodes.append(helper.make_node("Gather", ["X", "feat_idx"], ["xfeat"], axis=1))
        nodes.append(
            helper.make_node("GreaterOrEqual", ["thresholds", "xfeat"], ["ilb"])
        )
        nodes.append(helper.make_node("Cast", ["ilb"], ["ilk"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Sub", ["one", "ilk"], ["irk"]))

        # Scatter to full BFS width via selection matrices:
        #   QFULL[:, c]              = q of c's parent   (0 at the root)
        #   RFULL[:, c]              = route bit from parent into column c
        #   PFULL[:, c]              = p_js for internals, 1 for leaves
        gt_q = np.zeros((k, n_total), dtype=np.float32)
        gl = np.zeros((k, n_total), dtype=np.float32)
        gr = np.zeros((k, n_total), dtype=np.float32)
        gp = np.zeros((k, n_total), dtype=np.float32)
        row = 0
        for ti, t in enumerate(trees):
            off = base_rows[ti]
            idx = row + np.arange(t.n_internal)
            gt_q[idx, off + t.left_child_cols] = 1.0
            gt_q[idx, off + t.right_child_cols] = 1.0
            gl[idx, off + t.left_child_cols] = 1.0
            gr[idx, off + t.right_child_cols] = 1.0
            gp[idx, off + t.internal_cols] = 1.0
            row += t.n_internal
        add_init("gt_q", gt_q)
        add_init("gl", gl)
        add_init("gr", gr)
        add_init("gp", gp)
        nodes.append(helper.make_node("MatMul", ["qk", "gt_q"], ["qfull"]))
        nodes.append(helper.make_node("MatMul", ["ilk", "gl"], ["rlfull"]))
        nodes.append(helper.make_node("MatMul", ["irk", "gr"], ["rrfull"]))
        nodes.append(helper.make_node("Add", ["rlfull", "rrfull"], ["rfull"]))
        # PFULL = 1 - q @ gp: equals p_js at internal columns and 1 at leaf
        # columns (matching w_leaf = S_leaf).
        nodes.append(helper.make_node("MatMul", ["qk", "gp"], ["pint"]))
        nodes.append(helper.make_node("Sub", ["one", "pint"], ["pfull"]))
    # --- level recursions ---------------------------------------------------------
    w_blocks: list[str] = []
    for ti, t in enumerate(trees):
        tag0 = f"t{ti}l0"
        # Level 0: single root. Survival S = 1; weight = p_js(root), or the
        # plain root value when the tree has no splits at all (w_leaf = 1).
        add_init_i64(f"{tag0}a", [base_rows[ti]])
        add_init_i64(f"{tag0}b", [base_rows[ti] + 1])
        nodes.append(helper.make_node("Expand", ["one", "bshape"], [f"{tag0}_sq"]))
        if k:
            nodes.append(
                helper.make_node(
                    "Slice", ["pfull", f"{tag0}a", f"{tag0}b", "ax1"], [f"{tag0}_pw"]
                )
            )
            nodes.append(
                helper.make_node("Mul", [f"{tag0}_sq", f"{tag0}_pw"], [f"{tag0}_w"])
            )
        else:
            pass  # W_0 = S_0 = 1
        w_blocks.append(f"{tag0}_w")

        sp_name = ""  # survival factors of current level's internal nodes
        last_level = len(t.level_sizes) - 1
        for li, width in enumerate(t.level_sizes):
            tag = f"t{ti}l{li}"
            if li == 0:
                continue
            # Window of this level's BFS columns (tree-local + global offset).
            lo = int(t.level_offsets[li]) + base_rows[ti]
            hi = lo + width
            add_init_i64(f"{tag}s", [lo])
            add_init_i64(f"{tag}e", [hi])
            nodes.append(
                helper.make_node(
                    "Slice", ["qfull", f"{tag}s", f"{tag}e", "ax1"], [f"{tag}_qw"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Slice", ["rfull", f"{tag}s", f"{tag}e", "ax1"], [f"{tag}_rw"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Slice", ["pfull", f"{tag}s", f"{tag}e", "ax1"], [f"{tag}_pw"]
                )
            )
            # Duplicate parent-width survival factors per child pair.
            if li == 1:
                # Parents are just the root: S = 1.
                add_init_i64(f"{tag}wv", [width])
                nodes.append(
                    helper.make_node(
                        "Concat", ["bdim", f"{tag}wv"], [f"{tag}bw"], axis=0
                    )
                )
                nodes.append(
                    helper.make_node("Expand", ["one", f"{tag}bw"], [f"{tag}_sd"])
                )
            else:
                add_init_i64(f"{tag}dup", [-1, width])
                nodes.append(
                    helper.make_node("Unsqueeze", [sp_name, "ax2"], [f"{tag}_su"])
                )
                nodes.append(
                    helper.make_node(
                        "Concat", [f"{tag}_su", f"{tag}_su"], [f"{tag}_sc"], axis=2
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Reshape", [f"{tag}_sc", f"{tag}dup"], [f"{tag}_sd"]
                    )
                )
            # Survival into each level-li node (routing kills off-path nodes),
            # then its weight contribution.
            nodes.append(
                helper.make_node("Mul", [f"{tag}_sd", f"{tag}_qw"], [f"{tag}_sq1"])
            )
            nodes.append(
                helper.make_node("Mul", [f"{tag}_sq1", f"{tag}_rw"], [f"{tag}_sq"])
            )
            nodes.append(
                helper.make_node("Mul", [f"{tag}_sq", f"{tag}_pw"], [f"{tag}_w"])
            )
            w_blocks.append(f"{tag}_w")

            if li < last_level and t.internal_positions[li]:
                idx = np.array(t.internal_positions[li], dtype=np.int64)
                add_init_i64(f"{tag}gidx", idx)
                nodes.append(
                    helper.make_node(
                        "Gather", [f"{tag}_sq", f"{tag}gidx"], [f"{tag}_sp"], axis=1
                    )
                )
                sp_name = f"{tag}_sp"
            else:
                sp_name = ""

    if len(w_blocks) > 1:
        nodes.append(helper.make_node("Concat", w_blocks, ["wall"], axis=1))
        w_final = "wall"
    else:
        w_final = w_blocks[0]

    # --- outputs -----------------------------------------------------------------
    outputs = []
    if is_classification:
        # Average per-tree probability vectors: divide by the tree count.
        nodes.append(
            helper.make_node("MatMul", [w_final, "vals_mat"], ["probas_unnorm"])
        )
        nodes.append(
            helper.make_node("Mul", ["probas_unnorm", "inv_ntrees"], ["probabilities"])
        )
        nodes.append(
            helper.make_node("ArgMax", ["probabilities"], ["am"], axis=1, keepdims=0)
        )
        classes = np.asarray(getattr(estimator, "classes_", np.arange(n_classes)))
        if classes.shape[0] != n_classes:  # pragma: no cover
            classes = np.arange(n_classes)
        add_init_i64("classes", classes)
        nodes.append(helper.make_node("Gather", ["classes", "am"], ["labels"], axis=0))
        outputs = [
            helper.make_tensor_value_info("labels", TensorProto.INT64, [None]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, n_classes]
            ),
        ]
    else:
        nodes.append(helper.make_node("MatMul", [w_final, "vals_vec"], ["dot"]))
        nodes.append(helper.make_node("Mul", ["dot", "inv_ntrees"], ["pred2d"]))
        nodes.append(helper.make_node("Squeeze", ["pred2d", "ax1"], ["predictions"]))
        outputs = [
            helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None])
        ]

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    graph = helper.make_graph(
        nodes, f"{name}_graph", inputs, outputs, initializer=inits
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", target_opset)],
        producer_name=name,
        ir_version=8,
    )
    return model
