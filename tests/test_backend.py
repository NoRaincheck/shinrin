"""Tests for the native backend selection machinery (shinrin._backend)."""

from __future__ import annotations

import importlib

import pytest

from shinrin import _backend

NATIVE_SYMBOLS = [
    "Criterion",
    "MSE",
    "ClassificationCriterion",
    "Splitter",
    "BaseDenseSplitter",
    "MondrianSplitter",
    "Tree",
    "DepthFirstTreeBuilder",
    "PartialFitTreeBuilder",
    "DTYPE",
    "DOUBLE",
]


def test_default_backend_is_rust(monkeypatch):
    monkeypatch.delenv("SHINRIN_BACKEND", raising=False)
    assert _backend.get_backend() == "rust"


def test_invalid_backend_raises(monkeypatch):
    monkeypatch.setenv("SHINRIN_BACKEND", "fortran")
    with pytest.raises(ValueError, match="SHINRIN_BACKEND"):
        _backend.get_backend()


def test_backend_case_insensitive(monkeypatch):
    monkeypatch.setenv("SHINRIN_BACKEND", " MOJO ")
    assert _backend.get_backend() == "mojo"
    monkeypatch.setenv("SHINRIN_BACKEND", "Rust")
    assert _backend.get_backend() == "rust"


def test_rust_module_loads(monkeypatch):
    monkeypatch.setenv("SHINRIN_BACKEND", "rust")
    _backend._CACHE.clear()
    module = _backend.get_backend_module()
    assert module.__name__ == "shinrin._native"


def test_mojo_module_loads(monkeypatch):
    pytest.importorskip("mojo")
    monkeypatch.setenv("SHINRIN_BACKEND", "mojo")
    _backend._CACHE.clear()
    module = _backend.get_backend_module()
    assert module.__name__ == "shinrin._native_mojo"


def test_backends_expose_identical_symbols():
    pytest.importorskip("mojo")
    rust = importlib.import_module("shinrin._native")
    mojo = importlib.import_module("shinrin._native_mojo")
    for name in NATIVE_SYMBOLS:
        assert hasattr(rust, name), f"rust backend missing {name}"
        assert hasattr(mojo, name), f"mojo backend missing {name}"


def test_shims_use_active_backend(monkeypatch):
    pytest.importorskip("mojo")
    monkeypatch.setenv("SHINRIN_BACKEND", "mojo")
    _backend._CACHE.clear()
    import shinrin._skgarden.mondrian.tree._criterion as crit_shim
    import shinrin._skgarden.mondrian.tree._splitter as splitter_shim
    import shinrin._skgarden.mondrian.tree._tree as tree_shim

    for shim in (crit_shim, splitter_shim, tree_shim):
        importlib.reload(shim)
    try:
        assert tree_shim.Tree.__module__ == "shinrin._native_mojo"
        assert crit_shim.MSE.__module__ == "shinrin._native_mojo"
        assert splitter_shim.MondrianSplitter.__module__ == "shinrin._native_mojo"
    finally:
        monkeypatch.setenv("SHINRIN_BACKEND", "rust")
        importlib.reload(crit_shim)
        importlib.reload(splitter_shim)
        importlib.reload(tree_shim)
