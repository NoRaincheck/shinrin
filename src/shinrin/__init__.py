"""shinrin: random forests, quantile regression and rule extraction.

This package vendors scikit-garden (BSD-3) and skope-rules (BSD-3) so they
can be used without being installed as dependencies.

Optional dependencies:
    scikit-learn  – required for all tree/forest estimators
    pandas        – required for SkopeRules

Install optional dependencies with:
    pip install shinrin[sklearn]   – for tree/forest models
    pip install shinrin[pandas]    – for SkopeRules
    pip install shinrin[full]      – all optional dependencies
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeQuantileRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "ExtraTreesRegressor",
    "MondrianForestClassifier",
    "MondrianForestRegressor",
    "MondrianTreeClassifier",
    "MondrianTreeRegressor",
    "RandomForestQuantileRegressor",
    "RandomForestRegressor",
    "Rule",
    "SkopeRules",
    "TreeExplainer",
    "explanation",
    "replace_feature_name",
]

# ---------------------------------------------------------------------------
# Lazy-import machinery
# ---------------------------------------------------------------------------

# Mapping of public name → (module_path, attr_name)
_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "ExtraTreesRegressor": ("shinrin._skgarden.forest", "ExtraTreesRegressor"),
    "RandomForestRegressor": ("shinrin._skgarden.forest", "RandomForestRegressor"),
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
    "Rule": ("shinrin._skrules.rule", "Rule"),
    "replace_feature_name": ("shinrin._skrules.rule", "replace_feature_name"),
    "SkopeRules": ("shinrin._skrules.skope_rules", "SkopeRules"),
    "TreeExplainer": ("shinrin.shap", "TreeExplainer"),
    "explanation": ("shinrin.shap", "explanation"),
}

# Cache for resolved imports
_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy-import a symbol from the appropriate submodule."""
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
    """Include lazy-imported names in dir()."""
    return list(__all__)
