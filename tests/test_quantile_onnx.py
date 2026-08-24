"""Parity tests for ONNX export of quantile trees and forests."""

from __future__ import annotations

import numpy as np
import pytest

from shinrin import (
    DecisionTreeQuantileRegressor,
    ExtraTreeQuantileRegressor,
    ExtraTreesQuantileRegressor,
    RandomForestQuantileRegressor,
)
from shinrin.onnx import to_onnx

ONNX_INSTALLED = True
try:
    import onnx
    import onnxruntime  # noqa: F401
except ImportError:
    ONNX_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not ONNX_INSTALLED, reason="onnx / onnxruntime not installed"
)


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    Xtr = rng.randn(120, 5).astype(np.float32)
    ytr = (Xtr[:, 0] * 2.0 - Xtr[:, 1] + rng.randn(120) * 0.1).astype(np.float32)
    Xte = rng.randn(40, 5).astype(np.float32)
    return Xtr, ytr, Xte


def _predict_onnx(model, X, quantile):
    onx = to_onnx(model, X, quantile=quantile)
    onnx.checker.check_model(onx)
    import onnxruntime as ort

    sess = ort.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(["predictions"], {"X": X})[0]


QUANTILES = [0, 10, 50, 90, 100]


@pytest.mark.parametrize("q", QUANTILES)
@pytest.mark.parametrize(
    "Model", [DecisionTreeQuantileRegressor, ExtraTreeQuantileRegressor]
)
def test_quantile_tree_parity(regression_data, Model, q):
    Xtr, ytr, Xte = regression_data
    model = Model(random_state=0).fit(Xtr, ytr)
    got = _predict_onnx(model, Xte, q)
    expected = model.predict(Xte, quantile=q)
    assert np.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("q", QUANTILES)
@pytest.mark.parametrize(
    "Model", [RandomForestQuantileRegressor, ExtraTreesQuantileRegressor]
)
def test_quantile_forest_parity(regression_data, Model, q):
    Xtr, ytr, Xte = regression_data
    model = Model(n_estimators=5, random_state=0).fit(Xtr, ytr)
    got = _predict_onnx(model, Xte, q)
    expected = model.predict(Xte, quantile=q)
    assert np.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("q", [25, 50, 75])
def test_quantile_forest_ties(regression_data, q):
    """Ties in y must not change knot selection versus the reference."""
    Xtr, _, Xte = regression_data
    rng = np.random.RandomState(7)
    ytr = rng.randint(0, 4, len(Xtr)).astype(np.float32) * 2.5
    model = RandomForestQuantileRegressor(n_estimators=4, random_state=3).fit(Xtr, ytr)
    got = _predict_onnx(model, Xte, q)
    expected = model.predict(Xte, quantile=q)
    assert np.allclose(got, expected, atol=1e-4)


def test_quantile_forest_single_estimator(regression_data):
    Xtr, ytr, Xte = regression_data
    model = RandomForestQuantileRegressor(n_estimators=1, random_state=0).fit(Xtr, ytr)
    got = _predict_onnx(model, Xte, 50)
    assert np.allclose(got, model.predict(Xte, quantile=50), atol=1e-4)


@pytest.mark.parametrize("q", [None, -1, 101])
def test_quantile_requires_valid_fixed_q(regression_data, q):
    """Export bakes the quantile in: it must be given and lie in [0, 100]."""
    Xtr, ytr, _ = regression_data
    model = DecisionTreeQuantileRegressor(random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="(quantile|q should be)"):
        to_onnx(model, Xtr, quantile=q)
