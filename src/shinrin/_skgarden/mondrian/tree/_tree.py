"""
Shim for the native (Rust) tree implementation.

The classes and dtype constants are provided by the ``shinrin._native``
extension module (a Rust port of scikit-garden's ``_tree`` Cython module).
"""

from shinrin._native import DTYPE, DOUBLE, DepthFirstTreeBuilder, PartialFitTreeBuilder, Tree

__all__ = ["Tree", "DepthFirstTreeBuilder", "PartialFitTreeBuilder", "DTYPE", "DOUBLE"]
