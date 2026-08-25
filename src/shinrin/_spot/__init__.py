"""Vendored SPOT (SParse OpTimal) engine — optimal sparse decision trees.

Renamed from GOSDT; derived from https://github.com/ubc-systopia/gosdt-guesses
("Fast Sparse Decision Tree Optimization via Reference Ensembles").
"""

from .binarizer import NumericBinarizer
from .classifier import SPOTClassifier
from .status import Status
from .threshold_guessing import ThresholdGuessBinarizer
from .tree import Tree

__all__ = [
    "NumericBinarizer",
    "SPOTClassifier",
    "Status",
    "ThresholdGuessBinarizer",
    "Tree",
]
