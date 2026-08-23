"""Tests for the Metal (GPU) backend selection machinery.

These only exercise the Python-side resolver logic; they do not require a
GPU, a Metal toolchain, or prebuilt GPU extensions.
"""

from __future__ import annotations

import pytest

from shinrin._mlp._backend import get_mlp_backend
from shinrin._tabm._backend import get_tabm_backend


@pytest.mark.parametrize("module", ["mlp", "tabm"])
def test_metal_requires_library(monkeypatch, module):
    getter = get_mlp_backend if module == "mlp" else get_tabm_backend
    env = "SHINRIN_MLP_BACKEND" if module == "mlp" else "SHINRIN_TABM_BACKEND"
    monkeypatch.setenv(env, "metal")
    try:
        getter()
        # prebuilt GPU extensions exist in the source tree: selection succeeds
    except ImportError as exc:
        assert "build-mlp-metal" in str(exc) or "build-tabm-metal" in str(exc)
    finally:
        monkeypatch.delenv(env, raising=False)


@pytest.mark.parametrize("module", ["mlp", "tabm"])
def test_invalid_backend_rejects_metal_typos(monkeypatch, module):
    getter = get_mlp_backend if module == "mlp" else get_tabm_backend
    env = "SHINRIN_MLP_BACKEND" if module == "mlp" else "SHINRIN_TABM_BACKEND"
    monkeypatch.setenv(env, "metall")
    with pytest.raises(ValueError, match=env):
        getter()


def test_gpu_shared_lib_globs_are_exact():
    """The CPU resolver must not match GPU libraries and vice versa."""
    from shinrin._mlp._backend import _gpu_shared_lib as mlp_gpu
    from shinrin._mlp._backend import _mojo_shared_lib as mlp_cpu
    from shinrin._tabm._backend import _gpu_shared_lib as tabm_gpu
    from shinrin._tabm._backend import _mojo_shared_lib as tabm_cpu

    mlp_cpu_lib = mlp_cpu()
    tabm_cpu_lib = tabm_cpu()
    if mlp_cpu_lib is not None:
        assert "gpu" not in str(mlp_cpu_lib)
    if tabm_cpu_lib is not None:
        assert "gpu" not in str(tabm_cpu_lib)
    for fn in (mlp_gpu, tabm_gpu):
        lib = fn()
        if lib is not None:
            assert "gpu" in lib.name
