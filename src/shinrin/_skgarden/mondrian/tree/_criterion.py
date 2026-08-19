"""
Shim for the native (Rust) criterion implementation.

The classes are provided by the ``shinrin._native`` extension module (a Rust
port of scikit-garden's ``_criterion`` Cython module).
"""

from shinrin._native import ClassificationCriterion, Criterion, MSE

__all__ = ["Criterion", "MSE", "ClassificationCriterion"]
