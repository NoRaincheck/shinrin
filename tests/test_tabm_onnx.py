"""Tests for ONNX export of TabM models (parity via onnxruntime)."""

from __future__ import annotations

import numpy as np
import pytest

from shinrin import TabMClassifier, TabMRegressor

ONNX_INSTALLED = True
try:
    import onnx
except ImportError:
    ONNX_INSTALLED = False

ORT_INSTALLED = True
try:
    import onnxruntime as rt
except ImportError:
    ORT_INSTALLED = False

SKLEARN_INSTALLED = True
try:
    import sklearn  # noqa: F401
except ImportError:
    SKLEARN_INSTALLED = False

requires_onnx = pytest.mark.skipif(
    not (ONNX_INSTALLED and ORT_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx, onnxruntime and sklearn required",
)

ARCHS = ["tabm", "tabm-mini", "tabm-packed"]

# Small models so training stays fast; export correctness is what matters.
FIT_KWARGS = {
    "hidden_layer_sizes": (16,),
    "k": 4,
    "max_iter": 30,
    "early_stopping": False,
    "random_state": 42,
}


# ===================================================================
# Helpers
# ===================================================================


def _run_onnx(onnx_model, X):
    sess = rt.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in sess.get_outputs()]
    values = sess.run(None, {"X": np.ascontiguousarray(X, dtype=np.float32)})
    return dict(zip(names, values))


def _check_model(onnx_model) -> None:
    onnx.checker.check_model(onnx_model)


def _regression_data(n_train=300, n_test=100, n_features=6, seed=0):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_features)
    y_train = (
        np.sin(X_train[:, 0])
        + 0.5 * X_train[:, 1] ** 2
        - X_train[:, 2]
        + rng.randn(n_train) * 0.05
    )
    X_test = rng.randn(n_test, n_features)
    return X_train, y_train, X_test


def _binary_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(400, 5)
    y = (X[:, 0] + X[:, 1] - 0.5 * X[:, 2] > 0).astype(int)
    return X[:300], y[:300], X[300:]


def _multiclass_data(seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 4.0]])
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        X_parts.append(c + rng.randn(120, 2) * 0.9)
        y_parts.append(np.full(120, i))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(X))
    return X[perm][:280], y[perm][:280], X[perm][280:]


def _mixed_categorical_data(seed=0):
    """Two integer-coded categorical columns plus continuous features."""
    rng = np.random.RandomState(seed)
    n = 320
    cat0 = rng.randint(0, 3, size=n).astype(float)
    cat1 = rng.randint(0, 5, size=n).astype(float)
    num0 = rng.randn(n)
    num1 = rng.randn(n) * 2 + 1
    num2 = rng.randn(n) / 3
    cat_effect = np.where(cat0 == 0, 0.0, np.where(cat0 == 1, 1.0, -1.0))
    X = np.column_stack([cat0, cat1, num0, num1, num2])
    y = 2.0 * num0 - num1 + cat_effect + 0.5 * num2 + rng.randn(n) * 0.05
    return X[:260], y[:260], X[260:]


# ===================================================================
# Core parity: regressor / classifiers x architectures
# ===================================================================


@requires_onnx
class TestTabMRegressorExport:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_parity(self, arch):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _regression_data()
        model = TabMRegressor(arch_type=arch, **FIT_KWARGS)
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train, name=f"TabMReg_{arch}")
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test)

        assert set(out) == {"predictions"}
        assert out["predictions"].dtype == np.float32
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_parity_no_embeddings(self, arch):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _regression_data()
        model = TabMRegressor(arch_type=arch, use_embeddings=False, **FIT_KWARGS)
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        out = _run_onnx(onnx_model, X_test)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_parity_preproc_disabled(self, arch):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _regression_data()
        model = TabMRegressor(
            arch_type=arch,
            use_quantile=False,
            use_asinh=False,
            use_scaler=False,
            **FIT_KWARGS,
        )
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        out = _run_onnx(onnx_model, X_test)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )


@requires_onnx
class TestTabMClassifierExport:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_binary_parity(self, arch):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _binary_data()
        model = TabMClassifier(arch_type=arch, **FIT_KWARGS)
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test)

        assert set(out) == {"probabilities", "labels"}
        proba = out["probabilities"]
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(
            proba, model.predict_proba(X_test), rtol=1e-3, atol=1e-4
        )
        # labels output: integer indices into classes_
        np.testing.assert_array_equal(
            out["labels"], model.classes_[model.predict(X_test)]
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_multiclass_parity(self, arch):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _multiclass_data()
        model = TabMClassifier(arch_type=arch, **FIT_KWARGS)
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test)

        proba = out["probabilities"]
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-4)
        np.testing.assert_allclose(
            proba, model.predict_proba(X_test), rtol=1e-3, atol=1e-4
        )
        np.testing.assert_array_equal(out["labels"], model.predict(X_test))

    def test_string_class_names(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _multiclass_data(seed=3)
        y_train = np.asarray(["setosa", "versicolor", "virginica"])[y_train]
        model = TabMClassifier(**FIT_KWARGS)
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train, class_names=list(model.classes_))
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test)

        labels = out["labels"]
        assert labels.dtype.kind in ("U", "S", "O")
        decoded = np.array(
            [v.decode() if isinstance(v, bytes) else str(v) for v in labels]
        )
        expected = model.predict(X_test)
        np.testing.assert_array_equal(decoded, expected)


@requires_onnx
class TestTabMEdgeCases:
    def test_mixed_categorical_features(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _mixed_categorical_data()
        model = TabMRegressor(
            categorical_indices=[0, 1],
            **{**FIT_KWARGS, "max_iter": 40},
        )
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )

    def test_mixed_categorical_no_embeddings(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _mixed_categorical_data(seed=7)
        model = TabMRegressor(
            categorical_indices=[0, 1],
            use_embeddings=False,
            **FIT_KWARGS,
        )
        model.fit(X_train, y_train)

        onnx_model = to_onnx(model, X_train)
        out = _run_onnx(onnx_model, X_test)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )

    def test_constant_feature_single_bin(self):
        """A constant numerical column exercises the single-bin PLE path."""
        from shinrin.onnx import to_onnx

        X_train, y_train, X_test = _regression_data(n_train=250, n_test=60)
        # Force all columns numerical (no auto-categorical detection) ...
        kwargs = {**FIT_KWARGS, "categorical_cardinality_threshold": 0}
        model = TabMRegressor(use_quantile=False, **kwargs)
        # ... and inject a constant column.
        X_train_c = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_c = np.column_stack([X_test, np.ones(len(X_test))])
        model.fit(X_train_c, y_train)

        onnx_model = to_onnx(model, X_train_c)
        _check_model(onnx_model)
        out = _run_onnx(onnx_model, X_test_c)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test_c), rtol=1e-3, atol=1e-3
        )

    def test_dynamic_batch_size(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, _ = _regression_data(n_train=200, n_test=37)
        model = TabMRegressor(**FIT_KWARGS).fit(X_train, y_train)
        onnx_model = to_onnx(model, X_train[:50])

        for n in (1, 3, 37, 128):
            out = _run_onnx(onnx_model, X_train[:n])
            assert out["predictions"].shape == (n,)


@requires_onnx
class TestTabMExportAPI:
    def test_save_and_reload(self, tmp_path):
        import onnx

        from shinrin.onnx import save_onnx

        X_train, y_train, X_test = _regression_data(seed=11)
        model = TabMRegressor(**FIT_KWARGS).fit(X_train, y_train)

        path = tmp_path / "tabm.onnx"
        save_onnx(model, str(path), X=X_train)
        loaded = onnx.load(str(path))
        _check_model(loaded)
        out = _run_onnx(loaded, X_test)
        np.testing.assert_allclose(
            out["predictions"], model.predict(X_test), rtol=1e-3, atol=1e-3
        )

    def test_unfitted_raises(self):
        from shinrin.onnx import to_onnx

        with pytest.raises(ValueError, match="fitted"):
            to_onnx(TabMRegressor())

    def test_feature_count_mismatch_raises(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, _ = _regression_data(n_features=4)
        model = TabMRegressor(**FIT_KWARGS).fit(X_train, y_train)
        with pytest.raises(ValueError, match="feature"):
            to_onnx(model, np.zeros((5, 3)))

    def test_metadata_props(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, _ = _regression_data(seed=5)
        model = TabMRegressor(
            arch_type="tabm-mini",
            k=8,
            **{k: v for k, v in FIT_KWARGS.items() if k != "k"},
        ).fit(X_train, y_train)
        m = to_onnx(model, X_train, feature_names=[f"f{i}" for i in range(6)])
        props = {p.key: p.value for p in m.metadata_props}
        assert props["model_type"] == "TabMRegressor"
        assert props["arch_type"] == "tabm-mini"
        assert props["k"] == "8"
        assert props["feature_names"].startswith("f0,f1")

    def test_opset_too_low_raises(self):
        from shinrin.onnx import to_onnx

        X_train, y_train, _ = _regression_data(seed=5)
        model = TabMRegressor(**FIT_KWARGS).fit(X_train, y_train)
        with pytest.raises(ValueError, match="opset"):
            to_onnx(model, X_train, target_opset=9)
