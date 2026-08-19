"""
Shim for the native (Rust) splitter implementation.

The classes are provided by the ``shinrin._native`` extension module (a Rust
port of scikit-garden's ``_splitter`` Cython module).
"""

from shinrin._native import BaseDenseSplitter, MondrianSplitter, Splitter

__all__ = ["Splitter", "BaseDenseSplitter", "MondrianSplitter"]
