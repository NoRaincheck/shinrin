"""ONNX exporters for rule-based classifiers.

Four models share this module:

- :class:`shinrin.SkopeRules` — precision-weighted vote of threshold
  conjunctions; ``predict`` returns ``score > 0``.
- :class:`shinrin.CorelsClassifier` — certified-optimal ordered rule list
  over binary features; first matching rule wins, with a trailing default
  prediction. Antecedents are signed column ids (positive = feature true,
  negative = feature false).
- :class:`shinrin.OrdtClassifier` — skope-mined rule pool selected and
  ordered by CORELS; identical first-match semantics applied to boolean
  capture-matrix columns evaluated from raw features.
- :class:`shinrin.GOSDTClassifier` — optimal boolean decision tree
  (left child = feature true, right child = feature false); leaves carry a
  class index into ``classes_``.

Comparisons against mined thresholds run in float64 inside the graph,
matching the reference implementations (which compare float32 data against
float64-parsed literals). Outputs mirror each model's ``predict`` dtypes:
int64 labels for SkopeRules and class-index labels for GOSDT (plus its
one-hot ``probabilities``), and boolean labels for Corels and Ordt.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover
    TensorProto: Any = None
    helper: Any = None

DEFAULT_OPSET = 15


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_onnx() -> None:
    if helper is None or TensorProto is None:
        raise ImportError(
            "onnx is required for ONNX export. Install it with: pip install onnx"
        )


def _add_init(inits, nm, arr, dtype=TensorProto.FLOAT) -> None:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        # Preserve 0-d scalars (ascontiguousarray promotes to (1,)).
        inits.append(helper.make_tensor(nm, dtype, [], [arr.item()]))
    else:
        arr = np.ascontiguousarray(arr)
        inits.append(helper.make_tensor(nm, dtype, list(arr.shape), arr.ravel()))


def _add_scalar(inits, nm, value, dtype=TensorProto.DOUBLE) -> None:
    np_dtype = {TensorProto.DOUBLE: np.float64, TensorProto.FLOAT: np.float32}[dtype]
    _add_init(inits, nm, np.array(value, dtype=np_dtype), dtype)


def _make_model(nodes, inits, inputs, outputs, name: str, target_opset: int):
    graph = helper.make_graph(
        nodes, f"{name}_graph", inputs, outputs, initializer=inits
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", target_opset)],
        producer_name=name,
        ir_version=8,
    )


def _double_input(n_features: int) -> tuple[list, str]:
    """Declare the float ``X`` input plus its float64 view."""
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    return inputs, "xd"


def _to_double(nodes: list, n_features: int) -> str:
    nodes.append(helper.make_node("Cast", ["X"], ["xd"], to=TensorProto.DOUBLE))
    return "xd"


def _feature_column(nodes: list, inits: list, x: str, idx: int, cache: dict) -> str:
    """Gather one feature column ``(batch,)`` as a named tensor."""
    if idx not in cache:
        nm = f"f{idx}"
        _add_init(inits, f"{nm}_i", np.array(idx), TensorProto.INT64)
        nodes.append(helper.make_node("Gather", [x, f"{nm}_i"], [nm], axis=1))
        cache[idx] = nm
    return cache[idx]


def _parse_terms(rule: str) -> list[tuple[int, str, float]]:
    """Parse a skope-style query ("x3 <= 2.5 and x0 > 1") into triples."""
    from shinrin._ordt import _TERM_RE, _feat_index, _is_number

    terms: list[tuple[int, str, float]] = []
    for term in rule.split(" and "):
        match = _TERM_RE.match(term.strip())
        if match is None:
            raise ValueError(f"unparseable rule term: {term!r} in {rule!r}")
        name, op, rhs = match.groups()
        if op == "==" and not _is_number(rhs):
            continue  # degenerate "c == c" clause is always true
        terms.append((_feat_index(name), op, float(rhs)))
    return terms


def _rule_mask_node(
    nodes: list,
    inits: list,
    x: str,
    terms: list[tuple[int, str, float]],
    tag: str,
    col_cache: dict,
) -> str:
    """Emit a boolean ``(batch,)`` mask for one parsed rule."""
    if not terms:
        # Always-true rule: reuse any column in a tautology.
        col = _feature_column(nodes, inits, x, 0, col_cache)
        nodes.append(helper.make_node("Equal", [col, col], [tag]))
        return tag
    parts: list[str] = []
    for k, (feat, op, thr) in enumerate(terms):
        col = _feature_column(nodes, inits, x, feat, col_cache)
        _add_scalar(inits, f"{tag}t{k}", thr, TensorProto.DOUBLE)
        out = f"{tag}c{k}"
        op_node = {
            "<=": "LessOrEqual",
            ">": "Greater",
            "<": "Less",
            ">=": "GreaterOrEqual",
            "==": "Equal",
        }[op]
        nodes.append(helper.make_node(op_node, [col, f"{tag}t{k}"], [out]))
        parts.append(out)
    if len(parts) == 1:
        return parts[0]
    acc = parts[0]
    for k, part in enumerate(parts[1:]):
        out = f"{tag}a{k}"
        nodes.append(helper.make_node("And", [acc, part], [out]))
        acc = out
    return acc


# ---------------------------------------------------------------------------
# SkopeRules
# ---------------------------------------------------------------------------


def skope_rules_to_onnx(
    estimator,
    feature_names=None,
    name="ShinrinSkopeRules",
    target_opset=DEFAULT_OPSET,
):
    """Export a fitted SkopeRules detector.

    Output: ``labels`` int64 ``(batch,)`` equal to
    ``(decision_function(X) > 0).astype(int)``.
    """
    _require_onnx()
    n_features = int(estimator.n_features_)
    inputs, _ = _double_input(n_features)
    nodes: list = []
    inits: list = []
    x = _to_double(nodes, n_features)

    col_cache: dict = {}
    score = None
    for i, (query, perf) in enumerate(estimator.rules_without_feature_names_):
        mask = _rule_mask_node(nodes, inits, x, _parse_terms(query), f"r{i}", col_cache)
        nodes.append(
            helper.make_node("Cast", [mask], [f"r{i}f"], to=TensorProto.DOUBLE)
        )
        _add_scalar(inits, f"w{i}", float(perf[0]), TensorProto.DOUBLE)
        nodes.append(helper.make_node("Mul", [f"r{i}f", f"w{i}"], [f"r{i}s"]))
        if score is None:
            score = f"r{i}s"
        else:
            nodes.append(helper.make_node("Add", [score, f"r{i}s"], [f"s{i}"]))
            score = f"s{i}"
    if score is None:
        # No rules: every score is zero, so every prediction is 0.
        _add_init(inits, "ax0_i", np.array([0]), TensorProto.INT64)
        _add_init(inits, "zero_i64", np.array([0]), TensorProto.INT64)
        nodes.append(helper.make_node("Shape", ["X"], ["xs"]))
        nodes.append(helper.make_node("Gather", ["xs", "ax0_i"], ["bdim"], axis=0))
        nodes.append(helper.make_node("Expand", ["zero_i64", "bdim"], ["labels"]))
    else:
        _add_scalar(inits, "wzero", 0.0, TensorProto.DOUBLE)
        nodes.append(helper.make_node("Greater", [score, "wzero"], ["pos"]))
        nodes.append(
            helper.make_node("Cast", ["pos"], ["labels"], to=TensorProto.INT64)
        )
    outputs = [helper.make_tensor_value_info("labels", TensorProto.INT64, [None])]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)


# ---------------------------------------------------------------------------
# Corels
# ---------------------------------------------------------------------------


def _first_match_tail(
    nodes: list,
    inits: list,
    clauses: list[tuple[Any, bool]],
    default: bool,
    out_name: str,
) -> None:
    """Emit first-match rule-list logic.

    ``clauses`` is an ordered list of ``(match_tensor, negation_masked_tensors,
    prediction)`` handled as: walking the rules in reverse, each match
    overrides the running result. Implemented as a Mul chain:
    result = match * pred + (1 - match) * res.

    NOTE: Uses Mul-based logic instead of Where because ORT on macOS
    doesn't support BOOL->FLOAT Cast followed by Where.
    """
    _add_init(
        inits, "defpred", np.array(float(default), dtype=np.float32), TensorProto.FLOAT
    )
    _add_init(inits, "one_f", np.array(1.0, dtype=np.float32), TensorProto.FLOAT)
    res = "defpred"
    for i, (match, pred) in enumerate(reversed(clauses)):
        pred_f = f"pf{i}"
        _add_init(
            inits, pred_f, np.array(float(pred), dtype=np.float32), TensorProto.FLOAT
        )
        # Cast match (BOOL) to FLOAT
        match_f = f"mf{i}"
        nodes.append(helper.make_node("Cast", [match], [match_f], to=TensorProto.FLOAT))
        # result = match * pred + (1 - match) * res
        match_pred = f"mp{i}"
        nodes.append(helper.make_node("Mul", [match_f, pred_f], [match_pred]))
        one_minus_match = f"m1m{i}"
        nodes.append(helper.make_node("Sub", ["one_f", match_f], [one_minus_match]))
        unmatched = f"um{i}"
        nodes.append(helper.make_node("Mul", [one_minus_match, res], [unmatched]))
        out = f"rm{i}" if i < len(clauses) - 1 else out_name
        nodes.append(helper.make_node("Add", [match_pred, unmatched], [out]))
        res = out


def corels_to_onnx(
    estimator,
    feature_names=None,
    class_names=None,
    name="ShinrinCorels",
    target_opset=DEFAULT_OPSET,
):
    """Export a fitted CorelsClassifier.

    Output: ``labels`` BOOL ``(batch,)`` mirroring ``predict`` (which
    returns a boolean array).
    """
    _require_onnx()
    rl = estimator.rl_
    n_features = len(rl.features)
    rules = list(rl.rules)
    default = bool(rules[-1]["prediction"])

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    nodes: list = []
    inits: list = []
    # Binary features: nonzero counts as true after the model's uint8 cast;
    # >= 0.5 agrees for every in-domain (0/1) input.
    nodes.append(helper.make_node("Cast", ["X"], ["xf"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("GreaterOrEqual", ["xf", "half"], ["b"]))
    _add_scalar(inits, "half", 0.5, TensorProto.FLOAT)

    clauses: list = []
    for ri, entry in enumerate(rules[:-1]):
        conds: list[str] = []
        for ai, idx in enumerate(entry["antecedents"]):
            idx = int(idx)
            iu = abs(idx)
            if iu == 0:  # never matches outside the default clause
                break
            col = f"bc{iu - 1}"
            if f"bc{iu - 1}_i" not in {t.name for t in inits}:
                _add_init(inits, f"bc{iu - 1}_i", np.array(iu - 1), TensorProto.INT64)
                nodes.append(
                    helper.make_node("Gather", ["b", f"bc{iu - 1}_i"], [col], axis=1)
                )
            _add_init(inits, f"c{ri}_{ai}", np.array(idx > 0), TensorProto.BOOL)
            out = f"m{ri}_{ai}"
            nodes.append(helper.make_node("Equal", [col, f"c{ri}_{ai}"], [out]))
            conds.append(out)
        if not conds:
            match = "taut"
            if "taut" not in {n.output[0] for n in nodes}:
                nodes.append(helper.make_node("Equal", ["b", "b"], ["taut"]))
        elif len(conds) == 1:
            match = conds[0]
        else:
            acc = conds[0]
            for k, cond in enumerate(conds[1:]):
                out = f"a{ri}_{k}"
                nodes.append(helper.make_node("And", [acc, cond], [out]))
                acc = out
            match = acc
        clauses.append((match, bool(entry["prediction"])))

    if clauses:
        _first_match_tail(nodes, inits, clauses, default, "labels_f")
        # Cast FLOAT result back to BOOL for labels output
        nodes.append(
            helper.make_node("Cast", ["labels_f"], ["labels"], to=TensorProto.BOOL)
        )
    else:
        # Only the default prediction rule.
        _add_init(inits, "defpred", np.array([default]), TensorProto.BOOL)
        nodes.append(helper.make_node("Shape", ["X"], ["cs"]))
        _add_init(inits, "cax0", np.array([0]), TensorProto.INT64)
        nodes.append(helper.make_node("Gather", ["cs", "cax0"], ["cbdim"], axis=0))
        nodes.append(helper.make_node("Expand", ["defpred", "cbdim"], ["labels"]))
    outputs = [helper.make_tensor_value_info("labels", TensorProto.BOOL, [None])]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)


# ---------------------------------------------------------------------------
# OrdtClassifier
# ---------------------------------------------------------------------------


def ordt_to_onnx(
    estimator,
    feature_names=None,
    class_names=None,
    name="ShinrinOrdt",
    target_opset=DEFAULT_OPSET,
):
    """Export a fitted OrdtClassifier.

    Output: ``labels`` BOOL ``(batch,)`` mirroring ``predict``
    (first-match evaluation of the optimal rule list).
    """
    _require_onnx()
    n_features = int(estimator.n_features_)
    pool: list[str] = list(estimator.pool_rules_)
    entries = list(estimator.corels_.rl_.rules)
    default = bool(entries[-1]["prediction"])

    inputs, _ = _double_input(n_features)
    nodes: list = []
    inits: list = []
    x = _to_double(nodes, n_features)

    col_cache: dict = {}
    masks = [
        _rule_mask_node(nodes, inits, x, _parse_terms(r), f"z{i}", col_cache)
        for i, r in enumerate(pool)
    ]

    clauses: list = []
    for ri, entry in enumerate(entries[:-1]):
        conds: list[str] = []
        for ai, idx in enumerate(entry["antecedents"]):
            idx = int(idx)
            col = masks[abs(idx) - 1]
            if idx < 0:
                out = f"n{ri}_{ai}"
                nodes.append(helper.make_node("Not", [col], [out]))
                conds.append(out)
            else:
                conds.append(col)
        if len(conds) == 1:
            match = conds[0]
        else:
            acc = conds[0]
            for k, cond in enumerate(conds[1:]):
                out = f"oa{ri}_{k}"
                nodes.append(helper.make_node("And", [acc, cond], [out]))
                acc = out
            match = acc
        clauses.append((match, bool(entry["prediction"])))

    if clauses:
        _first_match_tail(nodes, inits, clauses, default, "labels_f")
        # Cast FLOAT result back to BOOL for labels output
        nodes.append(
            helper.make_node("Cast", ["labels_f"], ["labels"], to=TensorProto.BOOL)
        )
    else:
        _add_init(inits, "defpred", np.array([default]), TensorProto.BOOL)
        nodes.append(helper.make_node("Shape", ["X"], ["os"]))
        _add_init(inits, "oax0", np.array([0]), TensorProto.INT64)
        nodes.append(helper.make_node("Gather", ["os", "oax0"], ["obdim"], axis=0))
        nodes.append(helper.make_node("Expand", ["defpred", "obdim"], ["labels"]))
    outputs = [helper.make_tensor_value_info("labels", TensorProto.BOOL, [None])]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)


# ---------------------------------------------------------------------------
# GOSDT
# ---------------------------------------------------------------------------


def gosdt_to_onnx(
    estimator,
    feature_names=None,
    class_names=None,
    name="ShinrinGosdt",
    target_opset=DEFAULT_OPSET,
):
    """Export a fitted GOSDTClassifier (model 0).

    Outputs: ``labels`` with ``classes_`` dtype and ``probabilities``
    float ``(batch, n_classes)`` one-hot, mirroring ``predict`` /
    ``predict_proba``.
    """
    _require_onnx()
    tree = estimator.trees_[0]
    classes = np.asarray(tree.classes)
    n_classes = int(tree.n_classes)
    n_features = len(tree.features)

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])]
    nodes: list = []
    inits: list = []
    _add_init(inits, "one_f", np.array(1.0, dtype=np.float32), TensorProto.FLOAT)
    # Native predict casts X to bool: any nonzero value is true.
    nodes.append(helper.make_node("Cast", ["X"], ["b"], to=TensorProto.BOOL))

    def emit(node, prefix: str) -> str:
        from shinrin._spot.tree import Leaf, Node

        if isinstance(node, Leaf):
            _add_init(inits, f"{prefix}v", np.array(node.prediction), TensorProto.INT64)
            return f"{prefix}v"
        assert isinstance(node, Node)
        left = emit(node.left_child, f"{prefix}l")
        right = emit(node.right_child, f"{prefix}r")
        _add_init(inits, f"{prefix}fi", np.array(node.feature), TensorProto.INT64)
        col = f"{prefix}c"
        nodes.append(helper.make_node("Gather", ["b", f"{prefix}fi"], [col], axis=1))
        # Mul-based Where: result = col * left + (1 - col) * right
        # Cast BOOL col to FLOAT, left/right to FLOAT for Mul
        col_f = f"{prefix}cf"
        nodes.append(helper.make_node("Cast", [col], [col_f], to=TensorProto.FLOAT))
        left_f = f"{prefix}lf"
        nodes.append(helper.make_node("Cast", [left], [left_f], to=TensorProto.FLOAT))
        right_f = f"{prefix}rf"
        nodes.append(helper.make_node("Cast", [right], [right_f], to=TensorProto.FLOAT))
        matched_f = f"{prefix}mf"
        nodes.append(helper.make_node("Mul", [col_f, left_f], [matched_f]))
        unmatched_f = f"{prefix}uf"
        nodes.append(helper.make_node("Sub", ["one_f", col_f], [unmatched_f]))
        unmatched_f2 = f"{prefix}uf2"
        nodes.append(helper.make_node("Mul", [unmatched_f, right_f], [unmatched_f2]))
        out_f = f"{prefix}of"
        nodes.append(helper.make_node("Add", [matched_f, unmatched_f2], [out_f]))
        out = f"{prefix}v"
        nodes.append(helper.make_node("Cast", [out_f], [out], to=TensorProto.INT64))
        return out

    idx = emit(tree.tree, "t")

    labels_is_int = np.issubdtype(classes.dtype, np.number)
    if labels_is_int:
        _add_init(inits, "classes", classes.astype(np.int64), TensorProto.INT64)
        labels_dtype = TensorProto.INT64
    else:
        _add_init(
            inits,
            "classes",
            np.array([str(c) for c in classes], object),
            TensorProto.STRING,
        )
        labels_dtype = TensorProto.STRING
    nodes.append(helper.make_node("Gather", ["classes", idx], ["labels"], axis=0))

    _add_init(inits, "oh_depth", np.array([n_classes]), TensorProto.INT64)
    _add_init(inits, "oh_vals", np.array([0.0, 1.0], dtype=np.float32))
    nodes.append(
        helper.make_node("OneHot", [idx, "oh_depth", "oh_vals"], ["probabilities"])
    )
    outputs = [
        helper.make_tensor_value_info("labels", labels_dtype, [None]),
        helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, n_classes]
        ),
    ]
    return _make_model(nodes, inits, inputs, outputs, name, target_opset)
