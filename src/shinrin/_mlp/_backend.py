"""Backend resolver for the plain MLP trainer.

The MLP ships two trainers:

- ``numpy`` – pure NumPy reference implementation (always available)
- ``mojo`` – Mojo kernels compiled to ``shinrin._native_mlp``
  (build with ``just build-mlp-mojo``)

The backend is selected via the ``SHINRIN_MLP_BACKEND`` environment
variable (values: ``auto``, ``numpy``, ``mojo``). The default ``auto``
uses the Mojo kernels when a prebuilt library is present and falls back
to NumPy otherwise.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

__all__ = ["get_mlp_backend", "get_mlp_native"]

_CACHE: dict[str, Any] = {}
_MOJO_MODULE = "shinrin._native_mlp"


def get_mlp_backend() -> str:
    """Return the resolved backend name (``"numpy"`` or ``"mojo"``)."""
    raw = os.environ.get("SHINRIN_MLP_BACKEND", "auto").strip().lower()
    if raw not in ("auto", "numpy", "mojo"):
        raise ValueError(
            f"Invalid SHINRIN_MLP_BACKEND={raw!r}: expected 'auto', 'numpy' or 'mojo'"
        )
    if raw == "numpy":
        return "numpy"
    available = _mojo_shared_lib() is not None
    if raw == "mojo" and not available:
        raise ImportError(
            "SHINRIN_MLP_BACKEND=mojo requires a prebuilt "
            "shinrin/_native_mlp.so (run `just build-mlp-mojo`)"
        )
    return "mojo" if available else "numpy"


def _mojo_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_mlp*.so"))
    return candidates[0] if candidates else None


def get_mlp_native() -> Any:
    """Return the Mojo extension module for MLP kernels."""
    if "mojo" in _CACHE:
        return _CACHE["mojo"]
    if _mojo_shared_lib() is None:
        try:
            import mojo.importer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The Mojo MLP kernels require a prebuilt "
                "shinrin/_native_mlp.so (run `just build-mlp-mojo`) "
                "or the 'mojo' package for auto-compilation"
            ) from exc
    module = importlib.import_module(_MOJO_MODULE)
    _CACHE["mojo"] = module
    return module
