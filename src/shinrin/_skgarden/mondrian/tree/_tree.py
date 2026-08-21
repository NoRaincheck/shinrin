"""
Shim for the native tree implementation.

The classes and dtype constants are provided by the active backend
extension module (Rust port or Mojo port of scikit-garden's ``_tree``
Cython module), selected via ``shinrin._backend``.
"""

from shinrin._backend import get_backend_module

_native = get_backend_module()

DTYPE = _native.DTYPE
DOUBLE = _native.DOUBLE
DepthFirstTreeBuilder = _native.DepthFirstTreeBuilder
PartialFitTreeBuilder = _native.PartialFitTreeBuilder
Tree = _native.Tree

__all__ = ["Tree", "DepthFirstTreeBuilder", "PartialFitTreeBuilder", "DTYPE", "DOUBLE"]
