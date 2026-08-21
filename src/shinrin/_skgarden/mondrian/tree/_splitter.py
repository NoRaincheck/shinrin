"""
Shim for the native splitter implementation.

The classes are provided by the active backend extension module (a Rust
or Mojo port of scikit-garden's ``_splitter`` Cython module), selected
via ``shinrin._backend``.
"""

from shinrin._backend import get_backend_module

_native = get_backend_module()

BaseDenseSplitter = _native.BaseDenseSplitter
MondrianSplitter = _native.MondrianSplitter
Splitter = _native.Splitter

__all__ = ["Splitter", "BaseDenseSplitter", "MondrianSplitter"]
