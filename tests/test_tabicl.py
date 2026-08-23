"""Behaviour tests for the TabICL estimators (NumPy backend).

Uses tiny synthetic checkpoints (see ``tests/_tabicl_fixture.py``) so no
network access or optional torch install is required. The Mojo/torch parity
checks live in ``tests/test_tabicl_parity.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _tabicl_fixture import TINY_CLASSIFIER, TINY_REGRESSOR

from shinrin.tabicl import TabICLClassifier, TabICLRegressor


@pytest.fixture()
def clf_model_path(tmp_path):
    from _tabicl_fixture import write_synthetic_checkpoint

    write_synthetic_checkpoint(tmp_path, "tiny_clf", TINY_CLASSIFIER, seed=0)
    return tmp_path


@pytest.fixture()
def reg_model_path(tmp_path):
    from _tabicl_fixture import write_synthetic_checkpoint

    write_synthetic_checkpoint(tmp_path, "tiny_reg", TINY_REGRESSOR, seed=1)
    return tmp_path


def _make_clf(model_path, **kwargs):
    defaults: dict[str, Any] = {
        "backend": "numpy",
        "checkpoint_version": "tiny_clf.ckpt",
        "model_path": model_path,
        "allow_auto_download": False,
        "n_estimators": 2,
        "random_state": 0,
    }
    defaults.update(kwargs)
    return TabICLClassifier(**defaults)


def _make_reg(model_path, **kwargs):
    defaults: dict[str, Any] = {
        "backend": "numpy",
        "checkpoint_version": "tiny_reg.ckpt",
        "model_path": model_path,
        "allow_auto_download": False,
        "n_estimators": 2,
        "random_state": 0,
    }
    defaults.update(kwargs)
    return TabICLRegressor(**defaults)


def _dataset(n_train=48, n_test=12, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, 4) * 3
    y_train = rng.randint(0, n_classes, size=n_train)
    X_train = centers[y_train] + rng.randn(n_train, 4)
    y_test = rng.randint(0, n_classes, size=n_test)
    X_test = centers[y_test] + rng.randn(n_test, 4)
    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


# ---------------------------------------------------------------------------
# classifier API
# ---------------------------------------------------------------------------


def test_classifier_fit_predict_shapes(clf_model_path):
    X_train, y_train, X_test, _ = _dataset(n_classes=3)
    clf = _make_clf(clf_model_path).fit(X_train, y_train)
    assert list(clf.classes_) == [0, 1, 2]
    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    assert proba.shape == (len(X_test), 3)
    assert pred.shape == (len(X_test),)
    assert np.all(proba >= 0) and np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(pred)).issubset({0, 1, 2})


def test_classifier_deterministic_given_seed(clf_model_path):
    X_train, y_train, X_test, _ = _dataset(seed=1)
    p1 = (
        _make_clf(clf_model_path, random_state=7)
        .fit(X_train, y_train)
        .predict_proba(X_test)
    )
    p2 = (
        _make_clf(clf_model_path, random_state=7)
        .fit(X_train, y_train)
        .predict_proba(X_test)
    )
    np.testing.assert_allclose(p1, p2)


def test_classifier_score_pipeline(clf_model_path):
    """score() runs end to end on synthetic weights (accuracy itself is
    only meaningful against real checkpoints -- see the network-gated
    integration test below)."""
    X_train, y_train, X_test, y_test = _dataset(n_classes=3, seed=2)
    clf = _make_clf(clf_model_path).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert 0.0 <= score <= 1.0


def test_classifier_many_classes_hierarchical(clf_model_path):
    """More classes than ``max_classes`` routes through the hierarchical path."""
    n_classes = TINY_CLASSIFIER["max_classes"] + 3
    X_train, y_train, X_test, y_test = _dataset(n_classes=n_classes, seed=3)
    clf = _make_clf(clf_model_path).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), n_classes)
    assert clf.score(X_test, y_test) >= 1.0 / n_classes - 0.2


def test_classifier_rejects_too_many_classes_without_support(clf_model_path):
    X_train, y_train, _, _ = _dataset(n_classes=TINY_CLASSIFIER["max_classes"] + 2)
    with pytest.raises(ValueError, match="support_many_classes"):
        _make_clf(clf_model_path, support_many_classes=False).fit(X_train, y_train)


def test_classifier_kv_cache_matches_no_cache(clf_model_path):
    X_train, y_train, X_test, _ = _dataset(n_classes=3, seed=4)
    plain = _make_clf(clf_model_path).fit(X_train, y_train).predict_proba(X_test)
    cached = (
        _make_clf(clf_model_path, kv_cache=True)
        .fit(X_train, y_train)
        .predict_proba(X_test)
    )
    np.testing.assert_allclose(plain, cached, atol=5e-3)


def test_classifier_batch_size_does_not_change_predictions(clf_model_path):
    X_train, y_train, X_test, _ = _dataset(n_classes=3, seed=5)
    big = (
        _make_clf(clf_model_path, batch_size=64)
        .fit(X_train, y_train)
        .predict_proba(X_test)
    )
    small = (
        _make_clf(clf_model_path, batch_size=2)
        .fit(X_train, y_train)
        .predict_proba(X_test)
    )
    np.testing.assert_allclose(big, small, atol=1e-6)


def test_classifier_handles_nan_and_strings(clf_model_path):
    import pandas as pd

    rng = np.random.RandomState(6)
    X_train = rng.randn(40, 3)
    X_test = rng.randn(8, 3)
    X_train[0, 0] = np.nan
    X_test[1, 2] = np.nan
    y_train = (X_train[:, 0] > 0).astype(int)
    X_train_df = pd.DataFrame(
        {
            "a": X_train[:, 0],
            "b": X_train[:, 1],
            "c": ["x" if v > 0 else "y" for v in X_train[:, 2]],
        }
    )
    X_test_df = pd.DataFrame(
        {
            "a": X_test[:, 0],
            "b": X_test[:, 1],
            "c": ["x" if v > 0 else "y" for v in X_test[:, 2]],
        }
    )
    pred = _make_clf(clf_model_path).fit(X_train_df, y_train).predict(X_test_df)
    assert pred.shape == (8,)


def test_classifier_sklearn_integration(clf_model_path):
    from sklearn.model_selection import cross_val_score

    X_train, y_train, _, _ = _dataset(n_classes=3, seed=8)
    scores = cross_val_score(_make_clf(clf_model_path), X_train, y_train, cv=2)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0)


# ---------------------------------------------------------------------------
# regressor API
# ---------------------------------------------------------------------------


def test_regressor_mean_median_quantiles(reg_model_path):
    rng = np.random.RandomState(9)
    X = rng.randn(60, 3).astype(np.float32)
    y = X @ np.array([1.5, -2.0, 0.5]) + rng.randn(60) * 0.1
    reg = _make_reg(reg_model_path).fit(X, y)
    mean = reg.predict(X[:10])
    median = reg.predict(X[:10], output_type="median")
    quantiles = reg.predict(X[:10], output_type="quantiles", alphas=[0.25, 0.75])
    assert mean.shape == (10,)
    assert median.shape == (10,)
    assert quantiles.shape == (10, 2)
    assert np.all(quantiles[:, 0] <= quantiles[:, 1])


def test_regressor_output_type_dict(reg_model_path):
    rng = np.random.RandomState(10)
    X = rng.randn(40, 2).astype(np.float32)
    y = X.sum(axis=1)
    reg = _make_reg(reg_model_path).fit(X, y)
    out = reg.predict(X[:5], output_type=["mean", "quantiles"], alphas=[0.5])
    assert set(out.keys()) == {"mean", "quantiles"}
    assert out["quantiles"].shape == (5, 1)


def test_regressor_finite_outputs(reg_model_path):
    rng = np.random.RandomState(11)
    X = rng.randn(80, 3).astype(np.float32)
    y = X @ np.array([2.0, 1.0, -1.0]) + rng.randn(80) * 0.05
    reg = _make_reg(reg_model_path).fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (80,)
    assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# parameter validation / backend plumbing
# ---------------------------------------------------------------------------


def test_device_requires_torch_backend(clf_model_path):
    X, y = np.random.randn(20, 3), np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="device"):
        _make_clf(clf_model_path, device="cuda").fit(X, y)


def test_invalid_backend_env_raises(clf_model_path, monkeypatch):
    monkeypatch.setenv("SHINRIN_TABICL_BACKEND", "bogus")
    X, y = np.random.randn(20, 3), np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="SHINRIN_TABICL_BACKEND"):
        _make_clf(clf_model_path, backend=None).fit(X, y)


def _mojo_native_available() -> bool:
    try:
        from _tabicl_fixture import make_params

        from shinrin._tabicl._config import TabICLConfig
        from shinrin._tabicl._mojo_backend import TabICLMojoModel
    except ImportError:
        return False
    config = TabICLConfig.from_dict(TINY_CLASSIFIER)
    try:
        TabICLMojoModel(config, make_params(TINY_CLASSIFIER, seed=0))
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _mojo_native_available(),
    reason="native module not built (`just build-tabicl-mojo`)",
)
def test_classifier_many_classes_mojo_fallback(clf_model_path):
    """Mojo + more classes than max_classes warns and reloads torch/numpy."""
    n_classes = TINY_CLASSIFIER["max_classes"] + 3
    X_train, y_train, X_test, y_test = _dataset(n_classes=n_classes, seed=3)
    clf = _make_clf(clf_model_path, backend="mojo")
    with pytest.warns(UserWarning, match="falling back"):
        clf.fit(X_train, y_train)
    assert clf.backend_ in ("torch", "numpy")
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), n_classes)
    assert clf.score(X_test, y_test) >= 1.0 / n_classes - 0.2


def test_missing_checkpoint_no_download(tmp_path):
    X, y = np.random.randn(20, 3), np.zeros(20, dtype=int)
    clf = TabICLClassifier(
        backend="numpy",
        checkpoint_version="missing.ckpt",
        model_path=tmp_path,
        allow_auto_download=False,
    )
    with pytest.raises(FileNotFoundError):
        clf.fit(X, y)


def test_get_params_set_params_roundtrip(clf_model_path):
    clf = _make_clf(clf_model_path)
    params = clf.get_params()
    for name in ("n_estimators", "batch_size", "kv_cache", "backend"):
        assert name in params
    clf.set_params(batch_size=4, kv_cache=True)
    assert clf.batch_size == 4 and clf.kv_cache is True


def test_lazy_exports_available():
    import shinrin

    assert shinrin.TabICLClassifier is TabICLClassifier
    assert shinrin.TabICLRegressor is TabICLRegressor


# ---------------------------------------------------------------------------
# optional integration test (real checkpoint; requires network + torch)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_checkpoint_accuracy_sanity():
    """Non-degenerate accuracy on a separable task with the real v2 weights.

    Skipped unless ``SHINRIN_TABICL_INTEGRATION=1`` is set (downloads the
    ~110MB classifier checkpoint and needs torch for conversion).
    """
    import os

    if os.environ.get("SHINRIN_TABICL_INTEGRATION") != "1":
        pytest.skip("set SHINRIN_TABICL_INTEGRATION=1 to run the download test")
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch required for checkpoint conversion")

    X_train, y_train, X_test, y_test = _dataset(
        n_train=200, n_test=60, n_classes=4, seed=12
    )
    clf = TabICLClassifier(
        backend="numpy",
        n_estimators=8,
        random_state=0,
    ).fit(X_train, y_train)
    majority = max(np.bincount(y_test, minlength=4)) / len(y_test)
    assert clf.score(X_test, y_test) > majority
