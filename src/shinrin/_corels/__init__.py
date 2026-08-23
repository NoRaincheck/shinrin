"""Vendored pycorels (CORELS: Certifiably Optimal RulE ListS).

Derived from https://github.com/corels/pycorels (GPL-3.0). The Cython
extension has been replaced by bindings to the vendored C++ sources compiled
into ``shinrin._native`` without GMP. See LICENSE and NOTICE for provenance.
"""

from .corels import CorelsClassifier
from .utils import RuleList, load_from_csv

__all__ = ["CorelsClassifier", "load_from_csv", "RuleList"]
