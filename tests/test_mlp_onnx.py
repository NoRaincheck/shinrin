"""Tests for the MLP ONNX exporter.

Validates that exported ONNX models produce predictions matching the
native :class:`~shinrin.mlp.MLPClassifier` / :class:`~shinrin.mlp.MLPRegressor`
implementations.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from shinrin import MLPClassifier, MLPRegressor
from shinrin.onnx import to_onnx


def _run_onnx(model, X):
    """Run an ONNX model and return its outputs."""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"X": X.astype(np.float32)})


# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------


class TestMLPRegressor:
    @pytest.fixture(autouse=True)
    def _data(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 4).astype(np.float32)
        self.y = (
            self.X[:, 0] + 2 * self.X[:, 1] + 0.1 * rng.randn(200).astype(np.float32)
        )

    def _fit_regressor(self, **kwargs):
        return MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=100, **kwargs)

    def test_single_output(self):
        reg = self._fit_regressor(random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4), f"maxdiff={np.abs(got - exp).max()}"

    def test_multi_output(self):
        y2 = self.X[:, 2] + self.X[:, 3]
        reg = self._fit_regressor(random_state=0)
        reg.fit(self.X, np.column_stack([self.y, y2]).astype(np.float32))
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4), f"maxdiff={np.abs(got - exp).max()}"

    def test_activation_tanh(self):
        reg = self._fit_regressor(activation="tanh", random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4)

    def test_activation_logistic(self):
        reg = self._fit_regressor(activation="logistic", random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4)

    def test_activation_identity(self):
        reg = self._fit_regressor(activation="identity", random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4)

    def test_embeddings(self):
        reg = self._fit_regressor(use_embeddings=True, random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4)

    def test_quantization(self):
        reg = self._fit_regressor(quantization="ternary", random_state=0)
        reg.fit(self.X, self.y)
        model = to_onnx(reg, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])[0]
        exp = reg.predict(self.X[:20])
        assert np.allclose(got, exp, atol=1e-4)


# ---------------------------------------------------------------------------
# MLPClassifier — binary
# ---------------------------------------------------------------------------


class TestMLPClassifierBinary:
    @pytest.fixture(autouse=True)
    def _data(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 4).astype(np.float32)
        self.y = (self.X[:, 0] > 0).astype(int)

    def _fit_classifier(self, **kwargs):
        return MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=100, **kwargs)

    def test_labels_and_proba(self):
        clf = self._fit_classifier(random_state=0)
        clf.fit(self.X, self.y)
        model = to_onnx(clf, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])
        exp_labels = clf.predict(self.X[:20])
        exp_proba = clf.predict_proba(self.X[:20])
        assert np.allclose(got[0].ravel().astype(int), exp_labels)
        assert np.allclose(got[1], exp_proba, atol=1e-4)

    def test_activation_tanh(self):
        clf = self._fit_classifier(activation="tanh", random_state=0)
        clf.fit(self.X, self.y)
        model = to_onnx(clf, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])
        exp_proba = clf.predict_proba(self.X[:20])
        assert np.allclose(got[1], exp_proba, atol=1e-4)

    def test_embeddings(self):
        clf = self._fit_classifier(use_embeddings=True, random_state=0)
        clf.fit(self.X, self.y)
        model = to_onnx(clf, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])
        exp_proba = clf.predict_proba(self.X[:20])
        assert np.allclose(got[1], exp_proba, atol=1e-4)


# ---------------------------------------------------------------------------
# MLPClassifier — multiclass
# ---------------------------------------------------------------------------


class TestMLPClassifierMulti:
    @pytest.fixture(autouse=True)
    def _data(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(300, 4).astype(np.float32)
        self.y = np.array(
            [0 if x[0] + x[1] < 0 else (1 if x[0] - x[1] < 0 else 2) for x in self.X]
        )

    def _fit_classifier(self, **kwargs):
        return MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=100, **kwargs)

    def test_labels_and_proba(self):
        clf = self._fit_classifier(random_state=0)
        clf.fit(self.X, self.y)
        model = to_onnx(clf, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])
        exp_labels = clf.predict(self.X[:20])
        exp_proba = clf.predict_proba(self.X[:20])
        assert np.allclose(got[0].ravel().astype(int), exp_labels)
        assert np.allclose(got[1], exp_proba, atol=1e-4)

    def test_embeddings(self):
        clf = self._fit_classifier(use_embeddings=True, random_state=0)
        clf.fit(self.X, self.y)
        model = to_onnx(clf, self.X)
        onnx.checker.check_model(model)
        got = _run_onnx(model, self.X[:20])
        exp_proba = clf.predict_proba(self.X[:20])
        assert np.allclose(got[1], exp_proba, atol=1e-4)

    def test_categorical_features(self):
        X_cat = self.X.copy()
        X_cat[:, 2] = np.floor(self.X[:, 2] * 5) % 4  # 4 categories
        clf = self._fit_classifier(categorical_indices=[2], random_state=0)
        clf.fit(X_cat, self.y)
        model = to_onnx(clf, X_cat)
        onnx.checker.check_model(model)
        got = _run_onnx(model, X_cat[:20])
        exp_proba = clf.predict_proba(X_cat[:20])
        assert np.allclose(got[1], exp_proba, atol=1e-4)
