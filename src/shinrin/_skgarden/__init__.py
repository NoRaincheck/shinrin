from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.2"

__all__ = [
    "MondrianTreeClassifier",
    "MondrianTreeRegressor",
    "MondrianForestClassifier",
    "MondrianForestRegressor",
    "DecisionTreeQuantileRegressor",
    "ExtraTreesRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "RandomForestRegressor",
    "RandomForestQuantileRegressor",
]

_IMPORT_MAP = {
    "MondrianTreeClassifier": ("shinrin._skgarden.mondrian.tree.tree", "MondrianTreeClassifier"),
    "MondrianTreeRegressor": ("shinrin._skgarden.mondrian.tree.tree", "MondrianTreeRegressor"),
    "MondrianForestClassifier": ("shinrin._skgarden.mondrian.ensemble.forest", "MondrianForestClassifier"),
    "MondrianForestRegressor": ("shinrin._skgarden.mondrian.ensemble.forest", "MondrianForestRegressor"),
    "DecisionTreeQuantileRegressor": ("shinrin._skgarden.quantile.tree", "DecisionTreeQuantileRegressor"),
    "ExtraTreeQuantileRegressor": ("shinrin._skgarden.quantile.tree", "ExtraTreeQuantileRegressor"),
    "ExtraTreesQuantileRegressor": ("shinrin._skgarden.quantile.ensemble", "ExtraTreesQuantileRegressor"),
    "RandomForestQuantileRegressor": ("shinrin._skgarden.quantile.ensemble", "RandomForestQuantileRegressor"),
    "ExtraTreesRegressor": ("shinrin._skgarden.forest", "ExtraTreesRegressor"),
    "RandomForestRegressor": ("shinrin._skgarden.forest", "RandomForestRegressor"),
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
