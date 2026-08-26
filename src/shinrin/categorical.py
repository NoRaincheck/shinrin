"""Recover category-membership splits from target-encoded tree models.

Training a tree on target-encoded categorical columns replaces every
categorical split with an opaque ``x_encoded <= t`` comparison against a
smoothed statistic. :func:`to_categorical_tree` post-processes such a
model into :class:`CategoricalTree`, where each split that fell on a
categorical column carries the actual *set of categories* routed to the
true branch instead of the encoded threshold. This restores semantic
interpretability (``color in {red, green}``) and is the representation
consumed by the ONNX ``BRANCH_MEMBER`` export option (see
:func:`shinrin.to_onnx`).

The conversion is invertible: :meth:`CategoricalTree.to_encoded_thresholds`
maps member sets back to equivalent encoded thresholds, so either
representation can be derived from the other at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shinrin.onnx import _tree_iter


def _encoder_tables(
    encoder,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Validate a fitted encoder's duck-typed statistics protocol.

    Any object exposing ``categorical_features_``, ``categories_`` and
    ``encodings_`` (aligned per column, like :class:`shinrin.TargetEncoder`)
    works.
    """
    required = ("categorical_features_", "categories_", "encodings_")
    missing = [a for a in required if not hasattr(encoder, a)]
    if missing:
        raise TypeError(
            f"encoder {type(encoder).__name__} lacks fitted attribute(s) "
            f"{missing}; expected a shinrin TargetEncoder or compatible "
            "target encoder exposing categorical_features_/categories_/"
            "encodings_."
        )
    feats = [int(f) for f in encoder.categorical_features_]
    cats = {int(f): np.asarray(v) for f, v in encoder.categories_.items()}
    encs = {int(f): np.asarray(v) for f, v in encoder.encodings_.items()}
    return feats, cats, encs


@dataclass
class CategoricalTree:
    """Tree structure with categorical splits as explicit member sets.

    Mirrors the scikit-learn ``tree_`` array layout, except that splits on
    categorical columns are described by ``members[node]`` (the observed
    category codes routed to the true branch) rather than by an encoded
    threshold alone. Because ``threshold`` lives in the encoded space,
    this representation can also be consumed directly on raw category
    codes — no target encoder at inference time.

    Attributes
    ----------
    feature : ndarray of int, shape ``(n_nodes,)``
        Split column index per node (``-2`` at leaves).
    threshold : ndarray of float, shape ``(n_nodes,)``
        Split threshold in the *encoded* space (meaningless at leaves;
        kept verbatim so the conversion remains invertible).
    children_left : ndarray of int, shape ``(n_nodes,)``
        Left-child node index (``-1`` at leaves).
    children_right : ndarray of int, shape ``(n_nodes,)``
        Right-child node index (``-1`` at leaves).
    value : ndarray
        Leaf/statistics values copied from the source tree.
    members : list of ndarray or None
        Per node: sorted category codes taking the true branch for
        categorical splits, else ``None``.
    n_features_in : int
        Number of features the source model was fitted on.
    """

    feature: np.ndarray
    threshold: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    value: np.ndarray
    members: list[np.ndarray | None] = field(default_factory=list)
    n_features_in: int = 0

    @property
    def is_categorical(self) -> np.ndarray:
        """Boolean mask over nodes: True where the split is categorical."""
        return np.array([m is not None for m in self.members], dtype=bool)

    def apply(self, X) -> np.ndarray:
        """Route samples through the tree and return their leaf indices.

        Parameters
        ----------
        X : array-like of shape ``(n_samples, n_features_in)``
            Samples in the **original** (pre-target-encoding) convention:
            categorical columns hold raw category codes, which is exactly
            the input convention of an exported ``BRANCH_MEMBER`` graph.
            Routing by raw-code membership is equivalent to the source
            model comparing the *encoded* value against its threshold.

        Returns
        -------
        ndarray of int, shape ``(n_samples,)``
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        out = np.empty(X.shape[0], dtype=np.intp)
        for i, x in enumerate(X):
            node = 0
            while self.children_left[node] != -1:
                m = self.members[node]
                if m is not None:
                    go_left = x[self.feature[node]] in m
                else:
                    go_left = x[self.feature[node]] <= self.threshold[node]
                node = (
                    self.children_left[node] if go_left else self.children_right[node]
                )
            out[i] = node
        return out

    def to_encoded_thresholds(self, encoder) -> np.ndarray:
        """Return thresholds reproducing this tree's partitions.

        Inverse of :func:`to_categorical_tree`: recomputes each categorical
        split's threshold from its member set via
        ``threshold_for_partition`` and passes numeric thresholds through.
        The result equals the source tree's ``tree_.threshold`` up to
        floating-point identity whenever the member sets came from one.

        Parameters
        ----------
        encoder
            Fitted target encoder compatible with the one used during
            training (same statistics tables).

        Returns
        -------
        ndarray of float, shape ``(n_nodes,)``
        """
        _encoder_tables(encoder)  # validate protocol before use
        out = np.array(self.threshold, dtype=np.float64, copy=True)
        for i, m in enumerate(self.members):
            if m is None:
                continue
            f = int(self.feature[i])
            out[i] = encoder.threshold_for_partition(f, m)
        return out

    def to_text(self, feature_names=None) -> str:
        """Render the tree as indented ASCII rules.

        Parameters
        ----------
        feature_names : list of str, optional
            Names per column; defaults to ``x0, x1, ...``.

        Returns
        -------
        str
            Multi-line rendering, e.g. ``x1 in {0.0, 2.0}`` for a
            categorical split and ``x0 <= 2.50`` otherwise.
        """
        names = list(feature_names) if feature_names is not None else None
        lines: list[str] = []

        def fmt_value(v) -> str:
            arr = np.ravel(np.asarray(v, dtype=np.float64))
            return "[" + ", ".join(f"{x:.4g}" for x in arr) + "]"

        def walk(node: int, indent: str) -> None:
            if self.children_left[node] == -1:
                lines.append(f"{indent}value: {fmt_value(self.value[node])}")
                return
            fname = (
                names[int(self.feature[node])]
                if names is not None
                else f"x{int(self.feature[node])}"
            )
            m = self.members[node]
            cond = (
                f"{fname} in {{{', '.join(repr(float(c)) for c in m)}}}"
                if m is not None
                else f"{fname} <= {self.threshold[node]:.2f}"
            )
            lines.append(f"{indent}--- {cond}")
            walk(self.children_left[node], indent + "|   ")
            walk(self.children_right[node], indent + "|   ")

        walk(0, "")
        return "\n".join(lines)


def to_categorical_tree(model, encoder):
    """Convert a fitted tree/forest into :class:`CategoricalTree` form.

    Parameters
    ----------
    model : estimator
        Fitted single-tree model (exposes ``tree_``) or forest (exposes
        ``estimators_``), e.g. a shinrin Mondrian tree/forest trained on
        target-encoded features.
    encoder
        Fitted target encoder describing how categorical columns were
        mapped before training (:class:`shinrin.TargetEncoder` or any
        object exposing ``categorical_features_``, ``categories_`` and
        ``encodings_``).

    Returns
    -------
    CategoricalTree or list of CategoricalTree
        Single tree for tree models, one entry per forest member for
        ensembles.
    """
    feats, cats, encs = _encoder_tables(encoder)
    trees, _ = _tree_iter(model)
    results = []
    for tree in trees:
        n_nodes = len(tree.children_left)
        feature = np.asarray(tree.feature)
        threshold = np.asarray(tree.threshold, dtype=np.float64)
        members: list[np.ndarray | None] = []
        for i in range(n_nodes):
            f = int(feature[i])
            if tree.children_left[i] == -1 or f < 0:
                members.append(None)
            elif f in feats:
                thr = float(threshold[i])
                mask = encs[f] <= thr
                members.append(np.sort(cats[f][mask]))
            else:
                members.append(None)

        n_features = int(getattr(tree, "n_features", 0)) or int(
            getattr(model, "n_features_in_", 0)
        )
        results.append(
            CategoricalTree(
                feature=feature,
                threshold=threshold,
                children_left=np.asarray(tree.children_left),
                children_right=np.asarray(tree.children_right),
                value=np.asarray(tree.value),
                members=members,
                n_features_in=n_features,
            )
        )
    return results[0] if len(results) == 1 else results


__all__ = ["CategoricalTree", "to_categorical_tree"]
