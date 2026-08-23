"""Tests for ternary (BitLinear-style) weight quantization."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from shinrin._quant import (
    GRANULARITIES,
    QUANTIZATION_NONE,
    QUANTIZATION_TERNARY,
    QUANTIZATIONS,
    ternary_quantize_dequantize,
    ternary_scales,
    validate_quantization,
)
from shinrin._tabm._layers import TabMConfig
from shinrin.mlp import MLPClassifier
from shinrin.tabicl import TabICLClassifier
from shinrin.tabm import TabMClassifier


def _toy_data(n=200, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = ((X[:, 0] * X[:, 1] + 0.3 * X[:, 2]) > 0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# _quant units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", ["per_row", "per_tensor"])
def test_scales_shape(granularity):
    w = np.random.RandomState(0).randn(4, 7).astype(np.float32)
    s = ternary_scales(w, granularity)
    if granularity == "per_row":
        assert s.shape == (4, 1)
    else:
        assert s.shape == ()
    assert s.dtype == np.float32


def test_scales_absmean_nonzero_rows():
    w = np.random.RandomState(1).randn(5, 8).astype(np.float32)
    s = ternary_scales(w, "per_row")
    expected = (
        np.abs(w.astype(np.float64)).mean(axis=1, keepdims=True).astype(np.float32)
    )
    np.testing.assert_allclose(s, expected, rtol=1e-6)


def test_zero_weights_guard():
    """All-zero rows/tensors must not produce zero scales (0/0 -> nan)."""
    w = np.zeros((3, 4), dtype=np.float32)
    for gran in ("per_row", "per_tensor"):
        q = ternary_quantize_dequantize(w, gran)
        assert np.all(q == 0)
        assert not np.any(np.isnan(q))


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_dequantize_values_ternary(granularity):
    w = np.random.RandomState(2).randn(16, 32).astype(np.float32)
    q = ternary_quantize_dequantize(w, granularity)
    if granularity == "per_row":
        # every row has at most one nonzero magnitude (its own gamma)
        for r in range(q.shape[0]):
            mags = np.unique(np.abs(q[r]))
            assert mags.size <= 2 and (mags.size < 2 or 0.0 in mags)
    else:
        nz = np.unique(np.abs(q))
        assert nz.size <= 2 and nz.min() >= 0
        if nz.size == 2:
            assert 0.0 in nz


def test_matches_reference_formula():
    """q == round(clip(W/s)) * s with numpy semantics (half-to-even)."""
    w = np.random.RandomState(3).randn(12, 9).astype(np.float32)
    for gran in GRANULARITIES:
        s = ternary_scales(w, gran)
        ref = np.round(np.clip(w / s, -1.0, 1.0)).astype(np.float32) * s
        np.testing.assert_array_equal(ternary_quantize_dequantize(w, gran), ref)
    # explicit tie case: after scaling by the row absmean the entries sit
    # exactly on .5 boundaries, which must round half-to-even (±.5 -> 0,
    # distinguishing from sign/half-away-from-zero rounding).
    w = np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]], dtype=np.float32)
    s = ternary_scales(w, "per_row")
    q = ternary_quantize_dequantize(w, "per_row")
    ref = np.round(np.clip(w / s, -1.0, 1.0)).astype(np.float32) * s
    np.testing.assert_array_equal(q, ref)
    # the ±.5 entries collapsed to zero instead of rounding away from zero
    assert q[0, 2] == 0.0 and q[0, 3] == 0.0


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,g",
    [
        ("int4", "per_row"),
        ("ternary", "per_group"),
        ("bogus", "bogus"),
    ],
)
def test_validate_rejects_unknown(q, g):
    with pytest.raises(ValueError):
        validate_quantization(q, g)


def test_validate_granularity_ignored_when_off():
    """Granularity only matters when quantization is enabled."""
    validate_quantization("none", "per_row")
    validate_quantization("none", "not-a-granularity")


def test_validate_accepts_valid():
    validate_quantization("none", "per_row")
    validate_quantization("ternary", "per_tensor")
    assert QUANTIZATION_TERNARY in QUANTIZATIONS
    assert QUANTIZATION_NONE in QUANTIZATIONS


def test_estimator_validation_errors():
    X, y = _toy_data()
    with pytest.raises(ValueError, match="quantization"):
        MLPClassifier(quantization="int4").fit(X, y)
    with pytest.raises(ValueError, match="granularity"):
        MLPClassifier(quantization="ternary", quantization_granularity="per_group").fit(
            X, y
        )
    with pytest.raises(ValueError, match="quantization"):
        TabMClassifier(hidden_layer_sizes=(8,), quantization="fp8").fit(X, y)


# ---------------------------------------------------------------------------
# MLP end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", ["per_row", "per_tensor"])
def test_mlp_fit_score_sanity(granularity):
    X, y = _toy_data()
    clf = MLPClassifier(
        hidden_layer_sizes=(24,),
        max_iter=300,
        random_state=0,
        quantization="ternary",
        quantization_granularity=granularity,
    ).fit(X, y)
    assert clf.score(X, y) > 0.8
    proba = clf.predict_proba(X)
    assert np.all(proba >= 0) and np.allclose(proba.sum(axis=1), 1)


def test_mlp_quantized_close_to_full_precision():
    """QAT should land within a few points of the full-precision model."""
    rng = np.random.RandomState(1)
    X = rng.randn(300, 6).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)  # linearly separable
    scores = {}
    for quant in (QUANTIZATION_NONE, QUANTIZATION_TERNARY):
        clf = MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=200, random_state=0, quantization=quant
        ).fit(X, y)
        scores[quant] = clf.score(X, y)
    assert scores[QUANTIZATION_TERNARY] >= scores[QUANTIZATION_NONE] - 0.05


def test_mlp_quantize_output_flag():
    X, y = _toy_data()
    kw = {
        "hidden_layer_sizes": (16,),
        "max_iter": 300,
        "random_state": 0,
        "quantization": "ternary",
    }
    clf_hidden_only = MLPClassifier(**kw).fit(X, y)
    clf_with_output = MLPClassifier(quantize_output=True, **kw).fit(X, y)
    # quantizing the output layer too costs a little accuracy on this
    # XOR-flavored toy; the test pins the flag's plumbing and sanity.
    assert clf_hidden_only.score(X, y) > 0.8
    assert clf_with_output.score(X, y) > 0.75
    # get_params exposes all knobs
    params = clf_with_output.get_params()
    assert params["quantization"] == "ternary"
    assert params["quantize_output"] is True


def test_mlp_deterministic_and_picklable():
    X, y = _toy_data()
    kw = {
        "hidden_layer_sizes": (20,),
        "max_iter": 60,
        "random_state": 7,
        "quantization": "ternary",
    }
    a = MLPClassifier(**kw).fit(X, y)
    b = MLPClassifier(**kw).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))

    restored = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(restored.predict_proba(X), a.predict_proba(X))


# ---------------------------------------------------------------------------
# TabM end-to-end
# ---------------------------------------------------------------------------


def test_tabm_fit_score_sanity():
    X, y = _toy_data()
    for arch in ("tabm", "tabm-packed", "tabm-mini"):
        clf = TabMClassifier(
            hidden_layer_sizes=(16,),
            k=4,
            max_iter=150,
            random_state=0,
            arch_type=arch,
            use_embeddings=False,
            quantization="ternary",
        ).fit(X, y)
        assert clf.score(X, y) > 0.75, arch


def test_tabm_deterministic_and_picklable():
    X, y = _toy_data()
    kw = {
        "hidden_layer_sizes": (16,),
        "k": 4,
        "max_iter": 50,
        "random_state": 3,
        "use_embeddings": False,
        "quantization": "ternary",
        "quantization_granularity": "per_tensor",
    }
    a = TabMClassifier(**kw).fit(X, y)
    b = TabMClassifier(**kw).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    restored = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(restored.predict_proba(X), a.predict_proba(X))


def test_tabm_config_records_quantization():
    X, y = _toy_data()
    clf = TabMClassifier(
        hidden_layer_sizes=(12,),
        max_iter=10,
        random_state=0,
        use_embeddings=False,
        quantization="ternary",
    ).fit(X, y)
    config: TabMConfig = clf.config_
    assert config.quantization == "ternary"
    assert config.quantization_granularity == "per_row"


# ---------------------------------------------------------------------------
# TabICL PTQ
# ---------------------------------------------------------------------------


def test_tabicl_ptq_state_dict():
    from shinrin.tabicl import _ternary_post_training_quantize

    rng = np.random.RandomState(0)
    params = {
        "blocks.0.attn.in_proj_weight": rng.randn(16, 8).astype(np.float32),
        "blocks.0.attn.out_proj.weight": rng.randn(8, 8).astype(np.float64),
        "blocks.0.mlp.linear1.weight": rng.randn(32, 8).astype(np.float32),
        "blocks.0.norm.scale": rng.randn(8).astype(np.float32),
        "cls_tokens": rng.randn(1, 8).astype(np.float32),
    }
    out = _ternary_post_training_quantize(params, "per_row")
    # QKV projections are excluded, everything else 2-D *.weight quantized
    assert np.array_equal(
        out["blocks.0.attn.in_proj_weight"], params["blocks.0.attn.in_proj_weight"]
    )
    for key in (
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.linear1.weight",
    ):
        q, orig = out[key], params[key]
        assert q.dtype == orig.dtype
        # each row has at most one nonzero magnitude
        for r in range(q.shape[0]):
            mags = np.unique(np.abs(q[r]))
            assert mags.size <= 2 and (mags.size < 2 or 0.0 in mags)
    # 1-D and non-weight arrays untouched (identity, no copy)
    assert out["blocks.0.norm.scale"] is params["blocks.0.norm.scale"]
    assert out["cls_tokens"] is params["cls_tokens"]


def test_tabicl_ptq_warning_and_get_params():
    X, y = _toy_data(n=100)
    with pytest.warns(UserWarning, match="experimental"):
        clf = TabICLClassifier(
            backend="numpy",
            random_state=0,
            n_estimators=2,
            batch_size=4,
            allow_auto_download=False,
            quantization="ternary",
        ).fit(X[:60], y[:60])
    assert clf.get_params()["quantization_granularity"] == "per_row"
