from __future__ import annotations

import importlib
from typing import Any

__all__ = ['SkopeRules', 'Rule', 'replace_feature_name']

_IMPORT_MAP = {
    'SkopeRules': ('shinrin._skrules.skope_rules', 'SkopeRules'),
    'Rule': ('shinrin._skrules.rule', 'Rule'),
    'replace_feature_name': ('shinrin._skrules.rule', 'replace_feature_name'),
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
