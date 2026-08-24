"""SPOTSET — Sparse Optimal Rashomon Trees (vendored treeFARMS).

Enumerates the set of near-optimal sparse decision trees (the Rashomon set)
instead of a single optimum. Renamed from
https://github.com/ubc-systopia/treeFARMS, which itself builds on the same
gosdt-guesses lineage as :mod:`shinrin._spot`.
"""

from .classifier import SPOTSETClassifier
from .model_set import ModelSetContainer
from .tree_classifier import TreeClassifier

__all__ = [
    "ModelSetContainer",
    "SPOTSETClassifier",
    "TreeClassifier",
]
