"""Backend resolver for the native tree extension.

shinrin ships two implementations of the same native API (the classes and
constants used by ``shinrin._skgarden.mondrian.tree``):

- ``rust`` – the original maturin/pyo3 extension module ``shinrin._native``
- ``mojo`` – a Mojo port compiled to ``shinrin._native_mojo``

The backend is selected once per process via the ``SHINRIN_BACKEND``
environment variable (values: ``rust`` or ``mojo``). The default is
``rust`` so existing behavior never changes implicitly.

Example:
    SHINRIN_BACKEND=mojo python -c "import shinrin"
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

__all__ = ["get_backend", "get_backend_module"]

_CACHE: dict[str, Any] = {}

_RUST_MODULE = "shinrin._native"
_MOJO_MODULE = "shinrin._native_mojo"


def get_backend() -> str:
    """Return the configured backend name (``"rust"`` or ``"mojo"``)."""
    backend = os.environ.get("SHINRIN_BACKEND", "rust").strip().lower()
    if backend not in ("rust", "mojo"):
        raise ValueError(
            f"Invalid SHINRIN_BACKEND={backend!r}: expected 'rust' or 'mojo'"
        )
    return backend


def _mojo_shared_lib() -> Path | None:
    """Locate a prebuilt Mojo shared library, if any."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = sorted(pkg_dir.glob("_native_mojo*.so"))
    return candidates[0] if candidates else None


def _load_mojo() -> Any:
    """Load the Mojo extension module.

    Prefers a prebuilt shared library produced by ``just build-mojo``;
    falls back to ``mojo.importer`` auto-compilation from the ``.mojo``
    source when no library is present.
    """
    if _mojo_shared_lib() is None:
        try:
            import mojo.importer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SHINRIN_BACKEND=mojo requires the 'mojo' package "
                "(install with: pip install 'shinrin[mojo]') and either a "
                "prebuilt shinrin/_native_mojo.so (run `just build-mojo`) "
                "or mojo.importer for auto-compilation"
            ) from exc
    return importlib.import_module(_MOJO_MODULE)


def get_backend_module() -> Any:
    """Return the native extension module for the configured backend."""
    backend = get_backend()
    if backend in _CACHE:
        return _CACHE[backend]
    if backend == "rust":
        module = importlib.import_module(_RUST_MODULE)
    else:
        module = _load_mojo()
    _CACHE[backend] = module
    return module
