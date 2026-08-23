"""Backend resolver for the plain MLP trainer.

The MLP ships three trainers:

- ``numpy`` – pure NumPy reference implementation (always available)
- ``mojo`` – CPU Mojo kernels compiled to ``shinrin._native_mlp``
  (build with ``just build-mlp-mojo``)
- ``metal`` – Apple-GPU Mojo kernels compiled to
  ``shinrin._native_mlp_gpu`` (build with ``just build-mlp-metal``;
  requires the optional ``max`` package and Xcode's Metal toolchain)

The backend is selected via the ``SHINRIN_MLP_BACKEND`` environment
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

__all__ = ["get_mlp_backend", "get_mlp_native"]

_CACHE: dict[str, Any] = {}
_MOJO_MODULE = "shinrin._native_mlp"
_GPU_MODULE = "shinrin._native_mlp_gpu"
_BACKENDS = ("auto", "numpy", "mojo", "metal")


def get_mlp_backend() -> str:
    """Return the resolved backend name (``"numpy"``, ``"mojo"`` or ``"metal"``)."""
    raw = os.environ.get("SHINRIN_MLP_BACKEND", "auto").strip().lower()
    if raw not in _BACKENDS:
        raise ValueError(
            f"Invalid SHINRIN_MLP_BACKEND={raw!r}: expected one of {_BACKENDS}"
        )
    if raw == "numpy":
        return "numpy"
    if raw == "metal":
        if _gpu_shared_lib() is None:
            raise ImportError(
                "SHINRIN_MLP_BACKEND=metal requires a prebuilt "
                "shinrin/_native_mlp_gpu.so (run `just build-mlp-metal`, "
                "which needs the 'max' package and Xcode's Metal toolchain)"
            )
        return "metal"
    available = _mojo_shared_lib() is not None
    if raw == "mojo" and not available:
        raise ImportError(
            "SHINRIN_MLP_BACKEND=mojo requires a prebuilt "
            "shinrin/_native_mlp.so (run `just build-mlp-mojo`)"
        )
    return "mojo" if available else "numpy"


def _mojo_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_mlp.so"))
    return candidates[0] if candidates else None


def _gpu_shared_lib() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(pkg_dir.glob("_native_mlp_gpu*.so"))
    return candidates[0] if candidates else None


def get_mlp_native(backend: str | None = None) -> Any:
    """Return the native extension module for MLP kernels.

    ``backend`` defaults to the resolved value of ``get_mlp_backend()``;
    pass ``"mojo"`` or ``"metal"`` explicitly to pin a module.
    """
    if backend is None:
        backend = get_mlp_backend()
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
            build = "build-mlp-metal" if gpu else "build-mlp-mojo"
            extra = (
                " plus the 'max' package (`pip install shinrin[metal]`)" if gpu else ""
            )
            raise ImportError(
                f"The {backend} MLP kernels require a prebuilt "
                f"{module_name.replace('.', '/')}.so (run `just {build}`{extra}) "
                "or the 'mojo' package for auto-compilation"
            ) from exc
    module = importlib.import_module(module_name)
    _CACHE[backend] = module
    return module
