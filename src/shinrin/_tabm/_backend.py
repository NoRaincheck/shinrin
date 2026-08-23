"""Backend resolver for the TabM trainer.

TabM ships three trainers:

- ``numpy`` – pure NumPy reference implementation (always available)
- ``mojo`` – CPU Mojo kernels compiled to ``shinrin._native_tabm``
  (build with ``just build-tabm-mojo``)
- ``metal`` – Apple-GPU Mojo kernels compiled to
  ``shinrin._native_tabm_gpu`` (build with ``just build-tabm-metal``;
  requires the optional ``max`` package and Xcode's Metal toolchain)

The backend is selected via the ``SHINRIN_TABM_BACKEND`` environment
variable (values: ``auto``, ``numpy``, ``mojo``, ``metal``). The default
``auto`` uses the CPU Mojo kernels when a prebuilt library is present and
falls back to NumPy otherwise; Metal is strictly opt-in because it needs
extra dependencies and an Apple Silicon GPU.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

__all__ = ["get_tabm_backend", "get_tabm_native"]

_CACHE: dict[str, Any] = {}
_MOJO_MODULE = "shinrin._native_tabm"
_GPU_MODULE = "shinrin._native_tabm_gpu"
_BACKENDS = ("auto", "numpy", "mojo", "metal")


def get_tabm_backend() -> str:
    """Return the resolved backend name (``"numpy"``, ``"mojo"`` or ``"metal"``)."""
    raw = os.environ.get("SHINRIN_TABM_BACKEND", "auto").strip().lower()
    if raw not in _BACKENDS:
        raise ValueError(
            f"Invalid SHINRIN_TABM_BACKEND={raw!r}: expected one of {_BACKENDS}"
        )
    if raw == "numpy":
        return "numpy"
    if raw == "metal":
        if _gpu_shared_lib() is None:
            raise ImportError(
                "SHINRIN_TABM_BACKEND=metal requires a prebuilt "
                "shinrin/_native_tabm_gpu.so (run `just build-tabm-metal`, "
                "which needs the 'max' package and Xcode's Metal toolchain)"
            )
        return "metal"
    available = _mojo_shared_lib() is not None
    if raw == "mojo" and not available:
        raise ImportError(
            "SHINRIN_TABM_BACKEND=mojo requires a prebuilt "
            "shinrin/_native_tabm.so (run `just build-tabm-mojo`)"
        )
    return "mojo" if available else "numpy"


def _mojo_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_tabm.so"))
    return candidates[0] if candidates else None


def _gpu_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_tabm_gpu*.so"))
    return candidates[0] if candidates else None


def get_tabm_native(backend: str | None = None) -> Any:
    """Return the native extension module for TabM kernels.

    ``backend`` defaults to the resolved value of ``get_tabm_backend()``;
    pass ``"mojo"`` or ``"metal"`` explicitly to pin a module.
    """
    if backend is None:
        backend = get_tabm_backend()
    if backend == "numpy":
        raise ValueError("the numpy backend has no native module")
    if backend in _CACHE:
        return _CACHE[backend]
    gpu = backend == "metal"
    module_name = _GPU_MODULE if gpu else _MOJO_MODULE
    lib_missing = (_gpu_shared_lib() if gpu else _mojo_shared_lib()) is None
    if lib_missing:
        try:
            import mojo.importer  # noqa: F401
        except ImportError as exc:
            build = "build-tabm-metal" if gpu else "build-tabm-mojo"
            extra = (
                " plus the 'max' package (`pip install shinrin[metal]`)" if gpu else ""
            )
            raise ImportError(
                f"The {backend} TabM kernels require a prebuilt "
                f"{module_name.replace('.', '/')}.so (run `just {build}`{extra}) "
                "or the 'mojo' package for auto-compilation"
            ) from exc
    module = importlib.import_module(module_name)
    _CACHE[backend] = module
    return module
