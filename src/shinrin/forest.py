"""Random forest variants with an unbiased feature importances estimator.

Vendored and adapted from scikit-garden.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [  # noqa: F822
    "ExtraTreesRegressor",
    "RandomForestRegressor",
]

_IMPORT_MAP = {
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
