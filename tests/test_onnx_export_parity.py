"""Runtime-parity tests for shinrin ONNX exports.

These tests load every exported tree/forest/TabM graph into onnxruntime
and pin native-vs-ORT agreement, complementing the structural checks in
``test_onnx_benchmark.py`` and the round-trip import tests in
``test_onnx_import.py``.

All exported graphs accept float32 inputs regardless of the dtype used
for training; training with float64 data therefore also exercises how
much precision survives the float32 deployment path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import shinrin.onnx  # noqa: F401
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
    outputs = session.run(None, {inp.name: np.ascontiguousarray(X, dtype=np.float32)})
    return outputs


def _assert_close(actual, desired, atol):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).ravel(),
        np.asarray(desired, dtype=np.float64).ravel(),
        atol=atol,
        rtol=0,
    )


# Tolerances follow the conventions used across the repo's backend parity
# tests. The exact Mondrian export reproduces native predict to f32
# round-off; generic sklearn ensembles round thresholds/values to f32 in
# the graph, keeping agreement near 1e-6 for these scales.
F64_TOL = 1e-5
F32_TOL = 1e-5
TABM_TOL = 1e-3


@pytest.mark.skipif(not ORT_INSTALLED, reason="onnxruntime not installed")
class TestTreeForestOrtParity:
    """Native vs ONNX-runtime agreement for trees and forests."""

    @pytest.fixture(params=[np.float32, np.float64], ids=["f32", "f64"])
    def data(self, request):
        rng = np.random.RandomState(0)
        X = rng.randn(400, 6).astype(request.param)
        y_reg = (X @ rng.randn(6) + rng.randn(400) * 0.05).astype(request.param)
        y_bin = (X[:, 0] + 0.7 * X[:, 2] > 0).astype(int)
        y_multi = np.argmax(X[:, :3] + 0.05 * rng.randn(400, 3), axis=1)
        return X, y_reg, y_bin, y_multi

    def test_mondrian_tree_regressor(self, data):
        from shinrin import MondrianTreeRegressor

        X, y, _, _ = data
        model = MondrianTreeRegressor(max_depth=10, random_state=0).fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        # the Mondrian graph reproduces native predict exactly
        _assert_close(
            got, model.predict(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )

    def test_mondrian_forest_regressor(self, data):
        from shinrin import MondrianForestRegressor

        X, y, _, _ = data
        model = MondrianForestRegressor(n_estimators=8, max_depth=10, random_state=0)
        model.fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(
            got, model.predict(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )

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
        _assert_close(
            got, model.predict(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )

    def test_mondrian_tree_classifier_binary(self, data):
        from shinrin import MondrianTreeClassifier

        X, _, y, _ = data
        model = MondrianTreeClassifier(max_depth=10, random_state=0).fit(X, y)
        labels, proba = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(
            proba, model.predict_proba(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )
        assert (labels == model.predict(X)).all()

    def test_mondrian_forest_classifier_multiclass(self, data):
        from shinrin import MondrianForestClassifier

        X, _, _, y = data
        model = MondrianForestClassifier(n_estimators=8, max_depth=10, random_state=0)
        model.fit(X, y)
        labels, proba = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(
            proba, model.predict_proba(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )
        assert (labels == model.predict(X)).all()

    def test_sklearn_forest_classifier_multiclass(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, _, _, y = data
        model = sklearn.RandomForestClassifier(
            n_estimators=10, max_depth=10, random_state=0
        )
        model.fit(X, y)
        labels, proba = _ort_predict(to_onnx(model, X[:5]), X)
        _assert_close(
            proba, model.predict_proba(X), F32_TOL if X.dtype == np.float32 else F64_TOL
        )
        assert (labels == model.predict(X)).all()

    def test_gradient_boosting_regressor(self, data):
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, y, _, _ = data
        model = sklearn.GradientBoostingRegressor(
            n_estimators=15, max_depth=3, random_state=0
        )
        model.fit(X, y)
        got = _ort_predict(to_onnx(model, X[:5]), X)[0]
        _assert_close(got, model.predict(X), max(F64_TOL, 1e-4))

    def test_gradient_boosting_classifier_raises(self, data):
        # Boosting classifiers are regressor-trees internally; export must
        # refuse them explicitly instead of mis-detecting the task.
        sklearn = pytest.importorskip("sklearn.ensemble")
        X, _, y, _ = data
        model = sklearn.GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=0
        )
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match="GradientBoosting classifiers"):
            to_onnx(model, X[:5])


@pytest.mark.skipif(not ORT_INSTALLED, reason="onnxruntime not installed")
class TestMlpOrtParity:
    """Native vs ONNX-runtime agreement for MLP (float32 graphs)."""

    def test_binary_labels_match_probabilities_and_native(self):
        # Guards the binary label tail: deriving labels from a threshold on
        # the raw sigmoid column used to emit (n, 1) labels inconsistent
        # with the reported probabilities.
        try:
            from shinrin import MLPClassifier
        except ImportError:  # pragma: no cover
            pytest.skip("mlp dependencies missing")

        rng = np.random.RandomState(7)
        X = rng.randn(600, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=40, random_state=0)
        model.fit(X.astype(np.float32), y)

        labels, proba = _ort_predict(to_onnx(model, X[:5]), X)
        assert labels.ndim == 1
        assert (labels == proba.argmax(axis=1)).all()
        assert (labels == model.predict(X)).all()
        _assert_close(proba, model.predict_proba(X), TABM_TOL)


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
        got = _ort_predict(to_onnx(reg, X[:5]), X)[0]
        _assert_close(got, reg.predict(X), TABM_TOL)

        clf = TabMClassifier(hidden_layer_sizes=(16,), max_iter=20, random_state=0)
        clf.fit(X, c)
        proba = _ort_predict(to_onnx(clf, X[:5]), X)[0]
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
    for model in (tree, forest):
        proto = to_onnx(model, X[:3])
        onnx.checker.check_model(proto)


def _hard_tree_predict(model, X):
    """Plain decision-tree inference over stored arrays (no smoothing).

    Reference for the approximate ``tree-ensemble`` Mondrian export: leaf
    means averaged across trees, class counts normalized per leaf.
    """
    trees = [model] if hasattr(model, "tree_") else list(model.estimators_)
    outputs = []
    for est in trees:
        t = est.tree_ if hasattr(est, "tree_") else est
        out = np.empty((len(X), t.value.shape[-1]))
        for i, row in enumerate(X):
            nid = 0
            while t.children_left[nid] >= 0:
                go_left = row[t.feature[nid]] <= t.threshold[nid]
                nid = int(t.children_left[nid] if go_left else t.children_right[nid])
            out[i] = t.value[nid].ravel()
        if out.shape[1] > 1:
            sums = out.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            out = out / sums
        outputs.append(out)
    return np.stack(outputs).mean(axis=0)


class TestMondrianExportEncoding:
    """Export encoding follows the estimator's path_smoothing mode."""

    def test_default_constant_export_matches_native_even_huge(self):
        """Constant-mode forests export plain tree-ensembles that match."""
        from shinrin import MondrianForestRegressor
        from shinrin._mondrian_onnx import (
            MODE_TREE_ENSEMBLE,
            PROP_EXPORT_MODE,
            _collect_trees,
            _estimated_exact_bytes,
            mondrian_to_onnx,
        )

        rng = np.random.RandomState(4)
        X = rng.randn(4000, 20)
        y = X @ rng.randn(20) + rng.randn(4000) * 10
        model = MondrianForestRegressor(n_estimators=20, max_depth=16, random_state=0)
        model.fit(X.astype(np.float32), y)

        # The smoothing graph would exceed the protobuf limit here, but
        # constant-mode models never need it.
        assert _estimated_exact_bytes(_collect_trees(model)) > (2 << 30)
        assert model.path_smoothing is False

        # No fallback warning: this encoding is exact for the model.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proto = mondrian_to_onnx(model)
        assert not [w for w in caught if "falling back" in str(w.message)]

        props = {p.key: p.value for p in proto.metadata_props}
        assert props[PROP_EXPORT_MODE] == MODE_TREE_ENSEMBLE

        got = _ort_predict(proto, X[:500])[0]
        np.testing.assert_allclose(
            got,
            model.predict(X[:500].astype(np.float32)),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            got, _hard_tree_predict(model, X[:500]).ravel(), rtol=1e-4, atol=1e-4
        )

    def test_approximate_false_requires_smoothing_model(self):
        from shinrin import MondrianTreeRegressor
        from shinrin._mondrian_onnx import mondrian_to_onnx

        rng = np.random.RandomState(5)
        X = rng.randn(200, 4).astype(np.float32)
        y = X[:, 0]

        tree = MondrianTreeRegressor(max_depth=6, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="path_smoothing=False"):
            mondrian_to_onnx(tree, approximate=False)

    def test_small_models_default_to_tree_ensemble(self):
        from shinrin import MondrianForestRegressor, MondrianTreeRegressor
        from shinrin._mondrian_onnx import (
            MODE_EXACT,
            MODE_TREE_ENSEMBLE,
            PROP_EXPORT_MODE,
            mondrian_to_onnx,
        )

        rng = np.random.RandomState(5)
        X = rng.randn(200, 4).astype(np.float32)
        y = X[:, 0]

        # Default (constant) models always get the plain tree-ensemble.
        tree = MondrianTreeRegressor(max_depth=6, random_state=0).fit(X, y)
        props = {p.key: p.value for p in mondrian_to_onnx(tree).metadata_props}
        assert props[PROP_EXPORT_MODE] == MODE_TREE_ENSEMBLE

        forest = MondrianForestRegressor(n_estimators=3, max_depth=5, random_state=0)
        forest.fit(X, y)
        props = {p.key: p.value for p in mondrian_to_onnx(forest).metadata_props}
        assert props[PROP_EXPORT_MODE] == MODE_TREE_ENSEMBLE

        # Smoothing-mode models still get the exact graph by default.
        smooth_tree = MondrianTreeRegressor(
            max_depth=6, random_state=0, path_smoothing=True
        ).fit(X, y)
        props = {p.key: p.value for p in mondrian_to_onnx(smooth_tree).metadata_props}
        assert props[PROP_EXPORT_MODE] == MODE_EXACT

    def test_forced_approximate_classifier_matches_labels(self):
        from shinrin import MondrianForestClassifier
        from shinrin._mondrian_onnx import (
            MODE_TREE_ENSEMBLE,
            PROP_EXPORT_MODE,
            mondrian_to_onnx,
        )

        rng = np.random.RandomState(6)
        X = rng.randn(600, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = MondrianForestClassifier(n_estimators=6, max_depth=8, random_state=0)
        model.fit(X.astype(np.float32), y)

        proto = mondrian_to_onnx(model, approximate=True)
        props = {p.key: p.value for p in proto.metadata_props}
        assert props[PROP_EXPORT_MODE] == MODE_TREE_ENSEMBLE

        labels, proba = _ort_predict(proto, X)
        expected = _hard_tree_predict(model, X)
        _assert_close(proba, expected, 1e-5)
        assert (labels == expected.argmax(axis=1)).all()


def test_generic_forest_packs_single_tree_ensemble_node():
    """Generic RF/ET exports pack every tree into one TreeEnsemble node."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    from shinrin import MondrianForestRegressor

    rng = np.random.RandomState(3)
    X = rng.randn(150, 4).astype(np.float32)
    y = X[:, 0]

    # The Mondrian export for a smoothing-mode model is a self-contained
    # standard-domain graph: it reproduces native predict (path smoothing)
    # without any ai.onnx.ml operator.
    mondrian_forest = MondrianForestRegressor(
        n_estimators=4, max_depth=5, random_state=0, path_smoothing=True
    )
    mondrian_forest.fit(X, y)
    proto = to_onnx(mondrian_forest, X[:3])
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(proto)
    assert not [n for n in proto.graph.node if "TreeEnsemble" in n.op_type]

    # Default constant-mode models export as a plain TreeEnsemble node.
    default_forest = MondrianForestRegressor(
        n_estimators=4, max_depth=5, random_state=0
    )
    default_forest.fit(X, y)
    proto = to_onnx(default_forest, X[:3])
    onnx.checker.check_model(proto)
    te_nodes = [n for n in proto.graph.node if n.op_type == "TreeEnsembleRegressor"]
    assert len(te_nodes) == 1
    got = _ort_predict(proto, X)[0]
    np.testing.assert_allclose(got, default_forest.predict(X), rtol=1e-4, atol=1e-4)

    sklearn_forest = RandomForestRegressor(n_estimators=6, random_state=0).fit(X, y)
    proto = to_onnx(sklearn_forest, X[:3])
    te_nodes = [n for n in proto.graph.node if n.op_type == "TreeEnsembleRegressor"]
    assert len(te_nodes) == 1
