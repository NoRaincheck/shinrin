"""Runtime-parity tests for shinrin ONNX exports.

These tests load every exported tree/forest/TabM graph into onnxruntime
and pin native-vs-ORT agreement, complementing the structural checks in
``test_onnx_benchmark.py`` and the round-trip import tests in
``test_onnx_import.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

# NOTE: no global backend env overrides here - module-level
# ``os.environ.setdefault`` would run at collection time and silently
# change the backend other test modules (e.g. the MLP/bitlinear suites)
# train with.

from shinrin.onnx import to_onnx

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

ORT_INSTALLED = ort is not None


def _ort_predict(model_proto, X):
    assert ort is not None
    session = ort.InferenceSession(
        model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    dtype = np.float64 if "double" in inp.type else np.float32
    outputs = session.run(None, {inp.name: np.ascontiguousarray(X, dtype=dtype)})
    return outputs


def _assert_close(actual, desired, atol):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).ravel(),
        np.asarray(desired, dtype=np.float64).ravel(),
        atol=atol,
        rtol=0,
    )


# Tolerances follow the conventions used across the repo's backend parity
# tests. Mondrian backends compute internally at float32 even for float64
# input data, so their f64 tolerance reflects f32 round-off rather than
# machine precision.
F32_TOL = 1e-5
SKLEARN_F64_TOL = 1e-10
MONDRIAN_TOL = 1e-6
TABM_TOL = 1e-3


@pytest.mark.skipif(not ORT_INSTALLED, reason="onnxruntime not installed")
class TestTreeForestOrtParity:
    """Native vs ONNX-runtime agreement for trees and forests."""

    @pytest.fixture(params=[np.float32, np.float64], ids=["f32", "f64"])
    def data(self, request):
        rng = np.random.RandomState(0)
        X = rng.randn(400, 6).astype(request.param)
        y_reg = X @ rng.randn(6) + rng.randn(400) * 0.05
        y_bin = (X[:, 0] + 0.7 * X[:, 2] > 0).astype(int)
        y_multi = np.argmax(X[:, :3] + 0.05 * rng.randn(400, 3), axis=1)
        return X, y_reg, y_bin, y_multi

    def _tolerance(self, X, estimator_family):
        if X.dtype == np.float64 and estimator_family == "sklearn":
            return SKLEARN_F64_TOL
        return F32_TOL if X.dtype == np.float32 else MONDRIAN_TOL

    def test_mondrian_tree_regressor(self, data):
        from shinrin import MondrianTreeRegressor

        X, y, _, _ = data
        model = MondrianTreeRegressor(max_depth=10, random_state=0).fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(got, model.predict(X), self._tolerance(X, "mondrian"))

    def test_mondrian_forest_regressor(self, data):
        from shinrin import MondrianForestRegressor

        X, y, _, _ = data
        model = MondrianForestRegressor(n_estimators=8, max_depth=10, random_state=0)
        model.fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(got, model.predict(X), self._tolerance(X, "mondrian"))

    def test_sklearn_forest_regressor(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, y, _, _ = data
        model = sklearn.RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=0,
            criterion="squared_error",
        )
        model.fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(got, model.predict(X), self._tolerance(X, "sklearn"))

    def test_mondrian_tree_classifier_binary(self, data):
        from shinrin import MondrianTreeClassifier

        X, _, y, _ = data
        model = MondrianTreeClassifier(max_depth=10, random_state=0).fit(X, y)
        proba, labels = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(proba, model.predict_proba(X), self._tolerance(X, "mondrian"))
        assert (labels == model.predict(X)).all()

    def test_mondrian_forest_classifier_multiclass(self, data):
        from shinrin import MondrianForestClassifier

        X, _, _, y = data
        model = MondrianForestClassifier(n_estimators=8, max_depth=10, random_state=0)
        model.fit(X, y)
        proba, labels = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(proba, model.predict_proba(X), self._tolerance(X, "mondrian"))
        assert (labels == model.predict(X)).all()

    def test_sklearn_forest_classifier_multiclass(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, _, _, y = data
        model = sklearn.RandomForestClassifier(
            n_estimators=10, max_depth=10, random_state=0
        )
        model.fit(X, y)
        proba, labels = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(proba, model.predict_proba(X), self._tolerance(X, "sklearn"))
        assert (labels == model.predict(X)).all()

    def test_gradient_boosting_regressor(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, y, _, _ = data
        model = sklearn.GradientBoostingRegressor(
            n_estimators=15, max_depth=3, random_state=0
        )
        model.fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(got, model.predict(X), self._tolerance(X, "sklearn"))

    def test_gradient_boosting_classifier_binary(self, data):
        # Guards the prior-log-odds base handling: init_.predict returns a
        # class label for classifiers, not the raw-space constant.
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, _, y, _ = data
        model = sklearn.GradientBoostingClassifier(
            n_estimators=15, max_depth=3, random_state=0
        )
        model.fit(X, y)
        proba, labels = _ort_predict(to_onnx(model, X[:5]), X)
        tol = self._tolerance(X, "sklearn")
        _assert_close(proba, model.predict_proba(X), max(tol, 1e-7))
        assert (labels == model.predict(X)).all()

    def test_gradient_boosting_classifier_multiclass(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, _, _, y = data
        model = sklearn.GradientBoostingClassifier(
            n_estimators=15, max_depth=3, random_state=0
        )
        model.fit(X, y)
        proba, labels = _ort_predict(to_onnx(model, X[:5]), X)
        tol = self._tolerance(X, "sklearn")
        _assert_close(proba, model.predict_proba(X), max(tol, 1e-7))
        assert (labels == model.predict(X)).all()


@pytest.mark.skipif(not ORT_INSTALLED, reason="onnxruntime not installed")
class TestTabmOrtParity:
    """Native vs ONNX-runtime agreement for TabM (float32 graphs)."""

    def test_regressor_and_classifier(self):
        try:
            from shinrin import TabMClassifier, TabMRegressor
        except ImportError:
            pytest.skip("tabm dependencies missing")

        rng = np.random.RandomState(1)
        X = rng.randn(300, 8)
        y = X[:, 0] * 2.0 + rng.randn(300) * 0.01
        c = (X[:, 1] > 0).astype(int)

        reg = TabMRegressor(hidden_layer_sizes=(16,), max_iter=20, random_state=0)
        reg.fit(X, y)
        got = _ort_predict(to_onnx(reg, X[:5]), X.astype(np.float32))[0]
        _assert_close(got, reg.predict(X), TABM_TOL)

        clf = TabMClassifier(hidden_layer_sizes=(16,), max_iter=20, random_state=0)
        clf.fit(X, c)
        proba = _ort_predict(to_onnx(clf, X[:5]), X.astype(np.float32))[0]
        _assert_close(proba, clf.predict_proba(X), TABM_TOL)


def test_exports_are_valid_protos():
    onnx = pytest.importorskip("onnx")
    from shinrin import MondrianForestRegressor, MondrianTreeRegressor

    rng = np.random.RandomState(2)
    X = rng.randn(200, 4).astype(np.float32)
    y = X[:, 0]
    tree = MondrianTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    forest = MondrianForestRegressor(n_estimators=4, max_depth=5, random_state=0)
    forest.fit(X, y)
    for model, expected_trees in ((tree, 1), (forest, len(forest.estimators_))):
        proto = to_onnx(model, X[:3])
        onnx.checker.check_model(proto)
        # one TreeEnsemble node per tree
        n_nodes = sum(1 for n in proto.graph.node if n.op_type == "TreeEnsemble")
        assert n_nodes == expected_trees
