"""Canonical Mojo parameter-layout tests for TabICLv2.

The canonical layout in :mod:`shinrin._tabicl._mojo_layout` is the single
source of truth shared by the Python packer and the native kernel's offset
walk (``TabICLInference.__init__`` in ``_tabicl_kernels.mojo``). These tests
pin the spec against the synthetic fixture state dicts and verify that every
mismatch mode (missing / unknown / mis-shaped / reordered tensors) fails
loudly instead of silently misaligning weights.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _tabicl_fixture import TINY_CLASSIFIER, TINY_REGRESSOR, make_params

from shinrin._tabicl._config import TabICLConfig
from shinrin._tabicl._mojo_layout import (
    SSMAX_MLP,
    SSMAX_NONE,
    SSMAX_QAMLP_ELEMENTWISE,
    SSMAX_SCALES,
    canonical_tensor_specs,
    canonical_tensors,
    pack_params,
    ssmax_kind,
    total_elems,
)

# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


def _native_or_skip():
    from shinrin._tabicl._backend import get_tabicl_native

    try:
        return get_tabicl_native()
    except ImportError as exc:  # pragma: no cover - build not present
        pytest.skip(f"native TabICL kernels unavailable: {exc}")


# --------------------------------------------------------------------- #
# pure-Python layout checks (no native library required)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cfg_dict", [TINY_CLASSIFIER, TINY_REGRESSOR], ids=["classifier", "regressor"]
)
class TestLayoutBijection:
    def test_state_dict_matches_spec_exactly(self, cfg_dict: dict) -> None:
        cfg = TabICLConfig.from_dict(cfg_dict)
        params = make_params(cfg_dict)
        spec_names = {name for name, _ in canonical_tensor_specs(cfg)}
        assert spec_names == set(params)

    def test_pack_round_trip(self, cfg_dict: dict) -> None:
        cfg = TabICLConfig.from_dict(cfg_dict)
        params = make_params(cfg_dict)
        packed = pack_params(cfg, params)
        assert packed.size == total_elems(cfg)

        # Splitting the buffer back by the spec must recover each tensor.
        cur = 0
        for name, shape in canonical_tensor_specs(cfg):
            size = int(np.prod(shape, dtype=np.int64))
            chunk = packed[cur : cur + size].reshape(shape)
            np.testing.assert_array_equal(chunk, params[name])
            cur += size
        assert cur == packed.size

    def test_unique_names_and_positive_shapes(self, cfg_dict: dict) -> None:
        cfg = TabICLConfig.from_dict(cfg_dict)
        specs = canonical_tensor_specs(cfg)
        names = [name for name, _ in specs]
        assert len(names) == len(set(names))
        assert all(all(dim > 0 for dim in shape) for _, shape in specs)

    def test_canonical_tensors_mapping(self, cfg_dict: dict) -> None:
        cfg = TabICLConfig.from_dict(cfg_dict)
        assert canonical_tensors(cfg) == dict(canonical_tensor_specs(cfg))


def test_dims_array_contract_length() -> None:
    """dims indices 23-25 (ssmax kinds + col_affine) are part of the ABI."""
    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    dims = cfg.dims_array()
    assert dims.size == 26
    assert dims[23] == ssmax_kind("qassmax-mlp-elementwise")
    assert dims[24] == ssmax_kind("qassmax-mlp-elementwise")
    assert dims[25] == 0  # col_affine


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, SSMAX_NONE),
        (False, SSMAX_NONE),
        ("none", SSMAX_NONE),
        ("ssmax", SSMAX_SCALES),
        ("ssmax-mlp", SSMAX_MLP),
        ("qassmax-mlp-elementwise", SSMAX_QAMLP_ELEMENTWISE),
        (True, SSMAX_QAMLP_ELEMENTWISE),  # legacy bool -> default QA variant
    ],
)
def test_ssmax_kind_mapping(name, expected: int) -> None:
    assert ssmax_kind(name) == expected


def test_ssmax_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown ssmax variant"):
        ssmax_kind("turbo-max")


def test_pack_rejects_missing_tensor() -> None:
    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    params = make_params(TINY_CLASSIFIER)
    first_missing = canonical_tensor_specs(cfg)[0][0]
    del params[first_missing]
    with pytest.raises(KeyError, match="missing layout tensor"):
        pack_params(cfg, params)


def test_pack_rejects_unknown_tensor() -> None:
    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    params = make_params(TINY_CLASSIFIER)
    params["col_embedder.sneaky.weight"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="unknown to the layout"):
        pack_params(cfg, params)


def test_pack_rejects_shape_mismatch() -> None:
    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    params = make_params(TINY_CLASSIFIER)
    name, shape = canonical_tensor_specs(cfg)[0]
    params[name] = np.zeros(int(np.prod(shape)) + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        pack_params(cfg, params)


# --------------------------------------------------------------------- #
# native-side fail-fast checks (require the prebuilt shared library)
# --------------------------------------------------------------------- #


def test_native_fingerprint_matches_spec() -> None:
    """Kernel-walked offsets must equal offsets derived from the spec."""
    if os.environ.get("SHINRIN_TABICL_PARITY_MOJO") != "1":
        pytest.skip("set SHINRIN_TABICL_PARITY_MOJO=1 to run native kernel tests")
    native = _native_or_skip()
    from shinrin._tabicl._mojo_backend import expected_layout_offsets

    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    packed = pack_params(cfg, make_params(TINY_CLASSIFIER))
    handle = native.TabICLInference(cfg.dims_array(), packed)
    raw = handle.layout_offsets()
    if len(raw) == 1 and hasattr(raw[0], "__len__"):
        raw = raw[0]
    assert [int(x) for x in raw] == expected_layout_offsets(cfg)
    assert int(handle.param_count()) == packed.size


def test_model_construction_runs_full_validation() -> None:
    """End-to-end: canonical packing passes every native fail-fast gate."""
    if os.environ.get("SHINRIN_TABICL_PARITY_MOJO") != "1":
        pytest.skip("set SHINRIN_TABICL_PARITY_MOJO=1 to run native kernel tests")
    from shinrin._tabicl._mojo_backend import TabICLMojoModel

    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    model = TabICLMojoModel(cfg, make_params(TINY_CLASSIFIER))
    assert model.param_count == total_elems(cfg)


def test_native_rejects_short_buffer() -> None:
    """A truncated buffer must fail the kernel's length check."""
    if os.environ.get("SHINRIN_TABICL_PARITY_MOJO") != "1":
        pytest.skip("set SHINRIN_TABICL_PARITY_MOJO=1 to run native kernel tests")
    native = _native_or_skip()
    cfg = TabICLConfig.from_dict(TINY_CLASSIFIER)
    packed = pack_params(cfg, make_params(TINY_CLASSIFIER))
    with pytest.raises(ValueError, match="parameter buffer size mismatch"):
        native.TabICLInference(cfg.dims_array(), packed[:-1])
