"""
Shim for the native criterion implementation.

The classes are provided by the active backend extension module (a Rust
or Mojo port of scikit-garden's ``_criterion`` Cython module), selected
via ``shinrin._backend``.
"""

from shinrin._backend import get_backend_module

_native = get_backend_module()

ClassificationCriterion = _native.ClassificationCriterion
Criterion = _native.Criterion
MSE = _native.MSE

__all__ = ["Criterion", "MSE", "ClassificationCriterion"]
