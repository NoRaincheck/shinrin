"""Mondrian forests and trees (vendored from scikit-garden)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "MondrianForestClassifier",
    "MondrianForestRegressor",
    "MondrianTreeClassifier",
    "MondrianTreeRegressor",
]

_IMPORT_MAP = {
    "MondrianForestClassifier": (
        "shinrin._skgarden.mondrian.ensemble.forest",
        "MondrianForestClassifier",
    ),
    "MondrianForestRegressor": (
        "shinrin._skgarden.mondrian.ensemble.forest",
        "MondrianForestRegressor",
    ),
    "MondrianTreeClassifier": (
        "shinrin._skgarden.mondrian.tree.tree",
        "MondrianTreeClassifier",
    ),
    "MondrianTreeRegressor": (
        "shinrin._skgarden.mondrian.tree.tree",
        "MondrianTreeRegressor",
    ),
}

_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _IMPORT_MAP[name]
    if name in _cache:
        return _cache[name]
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    _cache[name] = obj
    return obj


def __dir__() -> list[str]:
    return list(__all__)
