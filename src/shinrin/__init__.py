"""shinrin: random forests, quantile regression and rule extraction.

This package vendors scikit-garden (BSD-3), skope-rules (BSD-3) and pycorels
(GPL-3) so they can be used without being installed as dependencies.

Optional dependencies:
    scikit-learn  – required for all tree/forest estimators
    pandas        – required for SkopeRules and OrdtClassifier

Install optional dependencies with:
    pip install shinrin[sklearn]   – for tree/forest models
    pip install shinrin[pandas]    – for SkopeRules / OrdtClassifier
    pip install shinrin[full]      – all optional dependencies
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "CorelsClassifier",
    "DecisionTreeQuantileRegressor",
    "ExtraTreeQuantileRegressor",
    "ExtraTreesQuantileRegressor",
    "ExtraTreesRegressor",
    "MLPClassifier",
    "MLPRegressor",
    "MondrianForestClassifier",
    "MondrianForestRegressor",
    "MondrianTreeClassifier",
    "MondrianTreeRegressor",
    "NumericBinarizer",
    "OrdtClassifier",
    "RandomForestQuantileRegressor",
    "RandomForestRegressor",
    "Rule",
    "SPOTClassifier",
    "SPOTSETClassifier",
    "SkopeRules",
    "TabICLClassifier",
    "TabICLRegressor",
    "TabMClassifier",
    "TabMRegressor",
    "ThresholdGuessBinarizer",
    "TreeExplainer",
    "benchmark_model_size",
    "benchmark_prediction",
    "benchmark_training",
    "explanation",
    "full_benchmark",
    "print_benchmark_report",
    "replace_feature_name",
    "save_onnx",
    "to_onnx",
]

# ---------------------------------------------------------------------------
# Lazy-import machinery
# ---------------------------------------------------------------------------

# Mapping of public name → (module_path, attr_name)
_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "CorelsClassifier": ("shinrin._corels.corels", "CorelsClassifier"),
    "NumericBinarizer": ("shinrin._spot.binarizer", "NumericBinarizer"),
    "OrdtClassifier": ("shinrin._ordt", "OrdtClassifier"),
    "ExtraTreesRegressor": ("shinrin._skgarden.forest", "ExtraTreesRegressor"),
    "MLPClassifier": ("shinrin.mlp", "MLPClassifier"),
    "MLPRegressor": ("shinrin.mlp", "MLPRegressor"),
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
    "SPOTClassifier": ("shinrin._spot.classifier", "SPOTClassifier"),
    "SPOTSETClassifier": ("shinrin._spotset.classifier", "SPOTSETClassifier"),
    "TabICLClassifier": ("shinrin.tabicl", "TabICLClassifier"),
    "TabICLRegressor": ("shinrin.tabicl", "TabICLRegressor"),
    "TabMClassifier": ("shinrin.tabm", "TabMClassifier"),
    "TabMRegressor": ("shinrin.tabm", "TabMRegressor"),
    "ThresholdGuessBinarizer": (
        "shinrin._spot.threshold_guessing",
        "ThresholdGuessBinarizer",
    ),
    "TreeExplainer": ("shinrin.shap", "TreeExplainer"),
    "explanation": ("shinrin.shap", "explanation"),
    "to_onnx": ("shinrin.onnx", "to_onnx"),
    "save_onnx": ("shinrin.onnx", "save_onnx"),
    "benchmark_training": ("shinrin.benchmark", "benchmark_training"),
    "benchmark_prediction": ("shinrin.benchmark", "benchmark_prediction"),
    "benchmark_model_size": ("shinrin.benchmark", "benchmark_model_size"),
    "full_benchmark": ("shinrin.benchmark", "full_benchmark"),
    "print_benchmark_report": ("shinrin.benchmark", "print_benchmark_report"),
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
