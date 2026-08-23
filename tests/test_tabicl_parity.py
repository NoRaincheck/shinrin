"""Numeric parity tests: torch <-> NumPy TabICL backends.

Both backends consume the *same* synthetic state dict (upstream torch
parameter names), so every stage must agree up to float32 accumulation
noise. The Mojo kernels are exercised as a construction/shape smoke test
only when explicitly enabled via ``SHINRIN_TABICL_PARITY_MOJO=1`` — they
are a performance scaffold and not yet held to numeric parity.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _tabicl_fixture import TINY_CLASSIFIER, TINY_REGRESSOR

from shinrin._tabicl._config import TabICLConfig
from shinrin.tabicl import CLASSIFIER_CHECKPOINT, REGRESSOR_CHECKPOINT

torch = pytest.importorskip("torch")

from shinrin._tabicl._model_numpy import TabICLNumPyModel
from shinrin._tabicl._model_torch import TabICLTorchModel


def _models(cfg_dict, seed=0):
    from _tabicl_fixture import make_params

    cfg_dict = {**cfg_dict}
    config = TabICLConfig.from_dict(cfg_dict)
    params = make_params(cfg_dict, seed=seed)
    return (
        TabICLTorchModel(config, params),
        TabICLNumPyModel(config, params),
        config,
    )


def _data(n_train=40, n_test=10, n_features=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_train + n_test, n_features).astype(np.float32)
    return X, n_train


TOL = {"rtol": 1e-3, "atol": 1e-4}


# ---------------------------------------------------------------------------
# stage-level parity (classifier weights)
# ---------------------------------------------------------------------------


def test_representations_parity():
    torch_m, numpy_m, _ = _models(TINY_CLASSIFIER)
    X, train_size = _data()
    y = np.random.RandomState(9).randint(0, TINY_CLASSIFIER["max_classes"], train_size)
    r_torch = torch_m.representations(X, y)
    r_np = numpy_m.representations(X, y)
    assert r_torch.shape == r_np.shape
    np.testing.assert_allclose(r_torch, r_np, rtol=TOL["rtol"], atol=TOL["atol"])


def test_predict_from_representations_logit_parity():
    torch_m, numpy_m, _ = _models(TINY_CLASSIFIER)
    X, train_size = _data()
    y = np.random.RandomState(9).randint(0, TINY_CLASSIFIER["max_classes"], train_size)
    # Feed the *same* row representations into both backends so this test
    # isolates the ICL stage from upstream float32 drift.
    r_torch = torch_m.representations(X, y)
    r_np = numpy_m.representations(X, y)
    np.testing.assert_allclose(r_torch, r_np, rtol=TOL["rtol"], atol=TOL["atol"])
    # Same buffer into both backends: isolates the ICL stage.
    out_torch = torch_m.predict_from_representations(r_torch, y)
    out_np = numpy_m.predict_from_representations(r_torch.copy(), y)
    n_test = X.shape[0] - train_size
    assert out_torch.shape == out_np.shape == (n_test, TINY_CLASSIFIER["max_classes"])
    np.testing.assert_allclose(out_torch, out_np, rtol=1e-2, atol=5e-3)

    # End-to-end from raw features (drift-amplified).
    e_torch = torch_m.forward(X, y)
    e_np = numpy_m.forward(X, y)
    np.testing.assert_allclose(e_torch, e_np, rtol=1e-2, atol=5e-2)


def test_predict_probability_parity():
    torch_m, numpy_m, _ = _models(TINY_CLASSIFIER)
    X, train_size = _data()
    y = np.random.RandomState(9).randint(0, TINY_CLASSIFIER["max_classes"], train_size)
    out_torch = torch_m.forward(X, y, return_logits=False)
    out_np = numpy_m.forward(X, y, return_logits=False)
    np.testing.assert_allclose(
        out_torch.sum(axis=-1), np.ones_like(out_torch[..., 0]), atol=1e-4
    )
    # End-to-end: small representation drift is amplified by the deep
    # encoder, so allow looser tolerance than the per-stage comparisons.
    np.testing.assert_allclose(out_torch, out_np, rtol=1e-2, atol=2e-2)


def test_kv_cache_parity_within_backend():
    """Cached and uncached forwards agree inside each backend."""
    from _tabicl_fixture import make_params

    config = TabICLConfig.from_dict(TINY_CLASSIFIER)
    params = make_params(TINY_CLASSIFIER, seed=0)
    X, train_size = _data()
    y = np.random.RandomState(9).randint(0, TINY_CLASSIFIER["max_classes"], train_size)

    for Model in (TabICLTorchModel, TabICLNumPyModel):
        model = Model(config, params)
        plain = model.forward(X, y)
        cache = model.build_cache(X[:train_size], y)
        cached = model.predict_with_cache(X[train_size:], cache)
        assert cached.shape == plain.shape
        np.testing.assert_allclose(plain, cached, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# many-class hierarchical parity (> max_classes)
# ---------------------------------------------------------------------------


def test_hierarchical_many_class_parity():
    cfg = {**TINY_CLASSIFIER}
    torch_m, numpy_m, _ = _models(cfg)
    rng = np.random.RandomState(3)
    n_train, n_test = 60, 12
    n_classes = cfg["max_classes"] + 3
    X = rng.randn(n_train + n_test, 5).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_train)

    r_torch = torch_m.representations(X, y)
    r_np = numpy_m.representations(X, y)
    p_torch = torch_m.predict_from_representations(r_torch, y, return_logits=False)
    p_np = numpy_m.predict_from_representations(r_np, y, return_logits=False)
    # Cross-check on identical inputs too (isolates the hierarchical path).
    p_iso_np = numpy_m.predict_from_representations(
        r_torch.copy(), y, return_logits=False
    )
    np.testing.assert_allclose(p_torch, p_iso_np, rtol=1e-2, atol=2e-2)
    assert p_torch.shape == p_np.shape == (n_test, n_classes)
    row_sums = p_np.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-3)


# ---------------------------------------------------------------------------
# regressor parity
# ---------------------------------------------------------------------------


def test_regressor_quantile_parity():
    torch_m, numpy_m, config = _models(TINY_REGRESSOR, seed=1)
    X, train_size = _data(seed=2)
    rng = np.random.RandomState(4)
    y = rng.randn(train_size).astype(np.float32)

    # Stage isolation on a shared representation buffer.
    r = np.random.RandomState(5).randn(1, X.shape[0], config.icl_dim).astype(np.float32)
    out_torch = torch_m.predict_from_representations(r, y)
    out_np = numpy_m.predict_from_representations(r, y)
    assert out_torch.shape[-1] == TINY_REGRESSOR["num_quantiles"]
    np.testing.assert_allclose(out_torch, out_np, rtol=1e-2, atol=5e-3)

    # End-to-end with looser tolerance (drift amplification).
    np.testing.assert_allclose(
        torch_m.forward(X, y), numpy_m.forward(X, y), rtol=1e-2, atol=2e-2
    )


# ---------------------------------------------------------------------------
# estimator-level parity through the public API (synthetic checkpoints)
# ---------------------------------------------------------------------------


def test_estimator_proba_parity(tmp_path):
    from _tabicl_fixture import write_synthetic_checkpoint

    from shinrin.tabicl import TabICLClassifier

    write_synthetic_checkpoint(tmp_path, "tiny_clf", TINY_CLASSIFIER, seed=0)
    rng = np.random.RandomState(7)
    X = rng.randn(50, 4).astype(np.float32)
    y = rng.randint(0, TINY_CLASSIFIER["max_classes"], size=50)

    probas = {}
    for backend in ("torch", "numpy"):
        clf = TabICLClassifier(
            backend=backend,
            checkpoint_version="tiny_clf.ckpt",
            model_path=tmp_path,
            allow_auto_download=False,
            n_estimators=2,
            random_state=0,
        ).fit(X, y)
        probas[backend] = clf.predict_proba(X[:9])
    np.testing.assert_allclose(probas["torch"], probas["numpy"], rtol=1e-2, atol=5e-3)


def test_estimator_default_checkpoint_names_match_plan():
    assert CLASSIFIER_CHECKPOINT.startswith("tabicl-classifier-v2")
    assert REGRESSOR_CHECKPOINT.startswith("tabicl-regressor-v2")


# ---------------------------------------------------------------------------
# Mojo kernels: numeric parity vs the torch reference (opt-in)
# ---------------------------------------------------------------------------


def _run_mojo_parity(cfg_dict: dict, y: np.ndarray) -> None:
    from shinrin._tabicl._mojo_backend import TabICLMojoModel

    try:
        from _tabicl_fixture import make_params
    except ImportError:
        pytest.skip("fixture helper unavailable")

    config = TabICLConfig.from_dict(cfg_dict)
    params = make_params(cfg_dict, seed=0)
    try:
        model = TabICLMojoModel(config, params)
    except ImportError:
        pytest.skip("native module not built")
    ref = TabICLTorchModel(config, params)

    rng = np.random.RandomState(11)
    n_train = y.shape[0]
    n_test = 7
    X = rng.randn(n_train + n_test, 5).astype(np.float32)

    rep_m = model.representations(X, y)
    rep_t = ref.representations(X, y)
    np.testing.assert_allclose(rep_m, rep_t, rtol=5e-4, atol=5e-5)

    logits_m = model.predict_from_representations(rep_m, y, return_logits=True)
    logits_t = ref.predict_from_representations(rep_t, y, return_logits=True)
    assert logits_m.shape == logits_t.shape
    np.testing.assert_allclose(logits_m, logits_t, rtol=5e-4, atol=5e-5)

    fwd_m = model.forward(X, y, return_logits=True)
    fwd_t = ref.forward(X, y, return_logits=True)
    assert fwd_m.shape == fwd_t.shape
    np.testing.assert_allclose(fwd_m, fwd_t, rtol=5e-4, atol=5e-5)

    # staged API must not mutate its inputs
    rep_copy = rep_m.copy()
    model.predict_from_representations(rep_m, y)
    np.testing.assert_array_equal(rep_m, rep_copy)


@pytest.mark.skipif(
    os.environ.get("SHINRIN_TABICL_PARITY_MOJO") != "1",
    reason="set SHINRIN_TABICL_PARITY_MOJO=1 with `just build-tabicl-mojo`",
)
def test_mojo_classifier_staged_parity():
    y = np.arange(12) % TINY_CLASSIFIER["max_classes"]
    _run_mojo_parity(TINY_CLASSIFIER, y)


@pytest.mark.skipif(
    os.environ.get("SHINRIN_TABICL_PARITY_MOJO") != "1",
    reason="set SHINRIN_TABICL_PARITY_MOJO=1 with `just build-tabicl-mojo`",
)
def test_mojo_regressor_staged_parity():
    rng = np.random.RandomState(3)
    _run_mojo_parity(TINY_REGRESSOR, rng.randn(12).astype(np.float32))
