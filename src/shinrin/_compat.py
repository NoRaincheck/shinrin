"""Compatibility layer for optional dependencies (scikit-learn, pandas).

All vendored modules (_skgarden, _skrules) import from this module instead of
directly from sklearn or pandas.  When the optional dependency is missing,
each imported name raises ImportError with a helpful message.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# scikit-learn shim
# ---------------------------------------------------------------------------

_sklearn_available: bool = False
_sklearn_error_msg: str = (
    "scikit-learn is required for this functionality. "
    "Install it with: pip install shinrin[sklearn]  (or pip install scikit-learn)"
)

_sklearn_modules: dict[str, ModuleType | None] = {}


def _import_sklearn_module(name: str) -> ModuleType:
    """Import a sklearn sub-module, caching the result."""
    mod = _sklearn_modules.get(name)
    if mod is not None:
        return mod
    try:
        mod = importlib.import_module(name)
        _sklearn_available = True
    except ImportError as exc:
        raise ImportError(_sklearn_error_msg) from exc
    _sklearn_modules[name] = mod
    return mod


def _get_sklearn(name: str) -> Any:
    """Lazy accessor for sklearn attributes."""
    # Map common dotted names to the underlying module
    if "." in name:
        mod_name, attr = name.rsplit(".", 1)
        return getattr(_import_sklearn_module(mod_name), attr)
    # Single-word names – try the top-level sklearn package
    return getattr(_import_sklearn_module("sklearn"), name)


# Pre-wire the most commonly requested dotted imports so that
# `from shinrin._compat import sklearn_ensemble` etc. works.
def _build_sklearn_proxy(module_name: str, proxy_name: str) -> None:
    """Return a proxy module that forwards all attribute access to *module_name*."""
    class _SklearnProxy(ModuleType):
        def __getattr__(self, name: str) -> Any:
            real_mod = _import_sklearn_module(module_name)
            return getattr(real_mod, name)

        def __dir__(self) -> list[str]:
            return list(_import_sklearn_module(module_name).__dict__.keys())

    sys.modules[f"shinrin._compat.{proxy_name}"] = _SklearnProxy(proxy_name)


# Register every sklearn submodule that the vendored code imports.
_SKLEARN_SUBMODULES = [
    ("sklearn.base", "sklearn_base"),
    ("sklearn.ensemble", "sklearn_ensemble"),
    ("sklearn.tree", "sklearn_tree"),
    ("sklearn.utils", "sklearn_utils"),
    ("sklearn.utils.validation", "sklearn_utils_validation"),
    ("sklearn.utils.multiclass", "sklearn_utils_multiclass"),
    ("sklearn.utils.class_weight", "sklearn_utils_class_weight"),
    ("sklearn.preprocessing", "sklearn_preprocessing"),
    ("sklearn.metrics", "sklearn_metrics"),
    ("sklearn.exceptions", "sklearn_exceptions"),
]

# Build proxy modules for all registered sklearn submodules
for _mod, _proxy in _SKLEARN_SUBMODULES:
    _build_sklearn_proxy(_mod, _proxy)


# ---------------------------------------------------------------------------
# pandas shim
# ---------------------------------------------------------------------------

_pandas_available: bool = False
_pandas_error_msg: str = (
    "pandas is required for SkopeRules. "
    "Install it with: pip install shinrin[pandas]  (or pip install pandas)"
)

_pandas_proxy: ModuleType | None = None


def _get_pandas() -> ModuleType:
    """Lazy accessor for the pandas module."""
    global _pandas_proxy
    if _pandas_proxy is not None:
        return _pandas_proxy
    try:
        _pandas_proxy = importlib.import_module("pandas")
        _pandas_available = True
    except ImportError as exc:
        raise ImportError(_pandas_error_msg) from exc
    return _pandas_proxy
