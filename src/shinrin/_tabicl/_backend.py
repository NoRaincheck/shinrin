"""Backend selection for TabICL estimators.

Backends are chosen via the ``backend`` constructor argument or the
``SHINRIN_TABICL_BACKEND`` environment variable (the constructor wins):

- ``"auto"``: first available among ``torch``, ``mojo``, ``numpy``
- ``"torch"``: own PyTorch implementation (needs the ``tabicl`` extra)
- ``"mojo"``: native Mojo inference kernels (needs a prebuilt
  ``shinrin/_native_tabicl.so``, run ``just build-tabicl-mojo``)
- ``"numpy"``: pure NumPy reference implementation

All backends load the same converted ``.npz`` weights (see
:mod:`shinrin._tabicl._checkpoint`) and are held to numeric parity by the
test suite.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

__all__ = ["get_tabicl_backend", "get_tabicl_native"]

VALID_BACKENDS = ("auto", "numpy", "torch", "mojo")

_CACHE: dict[str, Any] = {}
_MOJO_MODULE = "shinrin._native_tabicl"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _mojo_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_tabicl*.so"))
    return candidates[0] if candidates else None


def _mojo_available() -> bool:
    if _mojo_shared_lib() is not None:
        return True
    try:
        import mojo.importer  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(requested: str | None) -> str:
    """Resolve a user/backend request into an installed backend name.

    ``requested`` may be ``None`` (treated as ``"auto"``). Raises
    ``ValueError`` for unknown names and ``ImportError`` when a specific
    backend is requested but unavailable.
    """
    raw = requested or os.environ.get("SHINRIN_TABICL_BACKEND", "auto")
    raw = raw.strip().lower()
    if raw not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid SHINRIN_TABICL_BACKEND={raw!r}: expected one of "
            f"{', '.join(VALID_BACKENDS)}"
        )
    if raw == "numpy":
        return "numpy"
    if raw == "torch":
        if not _torch_available():
            raise ImportError(
                "TabICL backend 'torch' requires torch. Install it with "
                "`pip install shinrin[tabicl]`."
            )
        return "torch"
    if raw == "mojo":
        if not _mojo_available():
            raise ImportError(
                "TabICL backend 'mojo' requires a prebuilt shinrin/_native_tabicl.so "
                "(run `just build-tabicl-mojo`) or the 'mojo' package for "
                "auto-compilation."
            )
        return "mojo"
    # auto: prefer torch, then mojo, then numpy.
    if _torch_available():
        return "torch"
    if _mojo_available():
        return "mojo"
    return "numpy"


def get_tabicl_backend(requested: str | None = None) -> str:
    """Return the resolved backend name (cached for non-auto requests)."""
    key = requested or ""
    if key in _CACHE and requested is not None:
        return _CACHE[key]
    backend = resolve_backend(requested)
    if requested is not None:
        _CACHE[key] = backend
    return backend


def get_tabicl_native() -> Any:
    """Import and return the native Mojo TabICL module."""
    if "mojo" in _CACHE:
        return _CACHE["mojo"]
    if _mojo_shared_lib() is None:
        try:
            import mojo.importer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The Mojo TabICL kernels require a prebuilt "
                "shinrin/_native_tabicl.so (run `just build-tabicl-mojo`) "
                "or the 'mojo' package for auto-compilation"
            ) from exc
    module = importlib.import_module(_MOJO_MODULE)
    _CACHE["mojo"] = module
    return module
