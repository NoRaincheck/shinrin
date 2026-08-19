"""Quantile regression trees and forests (vendored from scikit-garden)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DecisionTreeQuantileRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "RandomForestQuantileRegressor",
]

_IMPORT_MAP = {
    "DecisionTreeQuantileRegressor": (
        "shinrin._skgarden.quantile.tree",
        "DecisionTreeQuantileRegressor",
    ),
    "ExtraTreeQuantileRegressor": (
        "shinrin._skgarden.quantile.tree",
        "ExtraTreeQuantileRegressor",
    ),
    "ExtraTreesQuantileRegressor": (
        "shinrin._skgarden.quantile.ensemble",
        "ExtraTreesQuantileRegressor",
    ),
    "RandomForestQuantileRegressor": (
        "shinrin._skgarden.quantile.ensemble",
        "RandomForestQuantileRegressor",
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
