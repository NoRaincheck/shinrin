"""Helpers for many-class classification (mixed radix + hierarchical tree).

These are pure NumPy/Python and shared by every backend. They mirror the
upstream TabICLv2 logic:

- ``compute_mixed_radix_bases`` / ``extract_mixed_radix_digit`` decompose
  labels into digits with balanced bases when the class count exceeds
  ``max_classes`` (used by the column embedding stage),
- ``group_classes`` / ``ClassNode`` / ``fit_hierarchical_tree`` build the
  balanced classification tree over training rows (used by the ICL stage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def compute_mixed_radix_bases(num_classes: int, max_classes: int) -> list[int]:
    """Balanced mixed-radix bases ``k_i <= max_classes`` with product >= num_classes."""
    if num_classes <= max_classes:
        return [num_classes]
    d = math.ceil(math.log(num_classes) / math.log(max_classes))
    k = min(math.ceil(num_classes ** (1.0 / d)), max_classes)
    bases = [k] * d
    product = k**d
    idx = 0
    while product < num_classes and idx < d:
        if bases[idx] < max_classes:
            product = product // bases[idx] * (bases[idx] + 1)
            bases[idx] += 1
        idx += 1
    return bases


def extract_mixed_radix_digit(
    y: np.ndarray, digit_idx: int, bases: list[int]
) -> np.ndarray:
    """Extract digit ``digit_idx`` from mixed-radix representation of labels."""
    divisor = 1
    for base in bases[digit_idx + 1 :]:
        divisor *= base
    return (y.astype(np.int64) // divisor) % bases[digit_idx]


def group_classes(num_classes: int, max_classes: int) -> tuple[np.ndarray, int]:
    """Partition classes into balanced groups of at most ``max_classes`` groups.

    Returns assignments (shape ``(num_classes,)``) and the group count.
    """
    if num_classes <= max_classes:
        return np.zeros(num_classes, dtype=np.int64), 1
    num_groups = min(math.ceil(num_classes / max_classes), max_classes)
    assignments = np.zeros(num_classes, dtype=np.int64)
    current_pos = 0
    remaining_classes = num_classes
    remaining_groups = num_groups
    for group in range(num_groups):
        size = math.ceil(remaining_classes / remaining_groups)
        assignments[current_pos : current_pos + size] = group
        current_pos += size
        remaining_classes -= size
        remaining_groups -= 1
    return assignments, num_groups


def label_encoding(y: np.ndarray) -> np.ndarray:
    """Remap target values to contiguous integers starting from 0."""
    unique_vals, inverse = np.unique(y, return_inverse=True)
    order = np.argsort(unique_vals)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(unique_vals))
    return remap[inverse].astype(np.int64)


@dataclass
class ClassNode:
    """Node in the hierarchical classification tree."""

    depth: int = 0
    classes_: np.ndarray | None = None
    is_leaf: bool = False
    R: np.ndarray | None = None
    y: np.ndarray | None = None
    group_indices: np.ndarray | None = None
    child_nodes: list[ClassNode] = field(default_factory=list)


def fit_hierarchical_tree(
    R_train: np.ndarray, y_train: np.ndarray, max_classes: int
) -> ClassNode:
    """Build the balanced hierarchy over training representations.

    Parameters
    ----------
    R_train : ndarray of shape (n_train, D)
        Row representations of the training data.
    y_train : ndarray of shape (n_train,)
        Integer class labels.
    max_classes : int
        Maximum number of classes handled natively per node.

    Returns
    -------
    ClassNode
        Root of the tree.
    """

    def fit_node(R: np.ndarray, y: np.ndarray, depth: int) -> ClassNode:
        node = ClassNode(depth=depth)
        unique_classes = np.unique(y)
        node.classes_ = unique_classes.astype(np.int64)
        node.R = R
        node.y = y
        if len(unique_classes) <= max_classes:
            node.is_leaf = True
            return node
        assignments, num_groups = group_classes(len(unique_classes), max_classes)
        mapping = {int(c): int(g) for c, g in zip(unique_classes, assignments)}
        node.group_indices = np.array([mapping[int(c)] for c in y], dtype=np.int64)
        for group in range(num_groups):
            mask = node.group_indices == group
            node.child_nodes.append(fit_node(R[mask], y[mask], depth + 1))
        return node

    return fit_node(R_train, y_train, 0)
