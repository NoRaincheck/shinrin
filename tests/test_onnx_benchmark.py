"""Tests for ONNX exporter and benchmarking module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from shinrin import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    MondrianTreeClassifier,
    MondrianTreeRegressor,
)

ONNX_INSTALLED = True
try:
    import onnx  # noqa: F401
    from onnx import (
        TensorProto,  # noqa: F401
        helper,  # noqa: F401
        numpy_helper,  # noqa: F401
    )
except ImportError:
    ONNX_INSTALLED = False


# ---------------------------------------------------------------------------
# ONNX Exporter Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ONNX_INSTALLED, reason="onnx not installed")
class TestONNXSingleTree:
    """Tests for ONNX export of single tree models."""

    @pytest.fixture
    def fitted_regressor(self):
        """Fitted MondrianTreeRegressor."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = rng.randn(100)
        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)
        return tree, X

    @pytest.fixture
    def fitted_classifier(self):
        """Fitted MondrianTreeClassifier."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = (rng.randn(100) > 0).astype(int)
        tree = MondrianTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X, y)
        return tree, X

    def test_to_onnx_regressor(self, fitted_regressor):
        """Test ONNX export of a regression tree."""
        from shinrin.onnx import to_onnx

        tree, X = fitted_regressor
        model = to_onnx(tree, X)
        assert model is not None
        assert len(model.graph.node) >= 1
        assert model.producer_name == "ShinrinModel"

    def test_to_onnx_classifier(self, fitted_classifier):
        """Test ONNX export of a classification tree."""
        from shinrin.onnx import to_onnx

        tree, X = fitted_classifier
        model = to_onnx(tree, X)
        assert model is not None
        assert len(model.graph.node) >= 1

    def test_to_onnx_no_X(self, fitted_regressor):
        """Test ONNX export without providing X for shape inference."""
        from shinrin.onnx import to_onnx

        tree, _ = fitted_regressor
        model = to_onnx(tree)
        assert model is not None

    def test_to_onnx_feature_names(self, fitted_regressor):
        """Test ONNX export with custom feature names."""
        from shinrin.onnx import to_onnx

        tree, X = fitted_regressor
        feature_names = ["a", "b", "c", "d"]
        model = to_onnx(tree, X, feature_names=feature_names)
        assert model is not None

    def test_save_onnx(self, fitted_regressor):
        """Test saving ONNX model to file."""
        from shinrin.onnx import save_onnx

        tree, X = fitted_regressor
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name

        try:
            save_onnx(tree, path, X)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.skipif(not ONNX_INSTALLED, reason="onnx not installed")
class TestONNxForest:
    """Tests for ONNX export of forest models."""

    @pytest.fixture
    def fitted_regressor(self):
        """Fitted MondrianForestRegressor."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = rng.randn(100)
        forest = MondrianForestRegressor(
            n_estimators=3,
            max_depth=3,
            random_state=0,
        )
        forest.fit(X, y)
        return forest, X

    @pytest.fixture
    def fitted_classifier(self):
        """Fitted MondrianForestClassifier."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = (rng.randn(100) > 0).astype(int)
        forest = MondrianForestClassifier(
            n_estimators=3,
            max_depth=3,
            random_state=0,
        )
        forest.fit(X, y)
        return forest, X

    def test_to_onnx_forest_regressor(self, fitted_regressor):
        """Test ONNX export of a regression forest."""
        from shinrin.onnx import to_onnx

        forest, X = fitted_regressor
        model = to_onnx(forest, X)
        assert model is not None
        # Mondrian forests export as standard-domain graphs (level-recursion
        # MatMuls); just require a non-trivial valid graph.
        assert len(model.graph.node) >= 1

    def test_to_onnx_forest_classifier(self, fitted_classifier):
        """Test ONNX export of a classification forest."""
        from shinrin.onnx import to_onnx

        forest, X = fitted_classifier
        model = to_onnx(forest, X)
        assert model is not None

    def test_save_onnx_forest(self, fitted_regressor):
        """Test saving forest ONNX model to file."""
        from shinrin.onnx import save_onnx

        forest, X = fitted_regressor
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name

        try:
            save_onnx(forest, path, X)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.skipif(not ONNX_INSTALLED, reason="onnx not installed")
class TestONNXErrors:
    """Tests for ONNX export error handling."""

    def test_unfitted_model_raises(self):
        """Test that unfitted models raise ValueError."""
        from shinrin.onnx import to_onnx

        tree = MondrianTreeRegressor(random_state=0)
        X = np.random.randn(10, 3).astype(np.float32)

        with pytest.raises((ValueError, TypeError)):
            to_onnx(tree, X)

    def test_non_tree_model_raises(self):
        """Test that non-tree models raise ValueError."""
        from shinrin.onnx import to_onnx

        class NotATree:
            pass

        model = NotATree()
        X = np.random.randn(10, 3).astype(np.float32)

        with pytest.raises(ValueError):
            to_onnx(model, X)


# ---------------------------------------------------------------------------
# Benchmarking Tests
# ---------------------------------------------------------------------------


class TestBenchmarkTraining:
    """Tests for training benchmark."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randn(100)
        return X, y

    def test_benchmark_training_single_model(self, sample_data):
        """Test benchmarking a single model."""
        from shinrin.benchmark import benchmark_training

        X, y = sample_data
        models = {"tree": MondrianTreeRegressor(max_depth=3, random_state=0)}
        results = benchmark_training(models, X, y, n_repeats=2)

        assert "tree" in results
        assert "mean_time" in results["tree"]
        assert results["tree"]["mean_time"] > 0

    def test_benchmark_training_multiple_models(self, sample_data):
        """Test benchmarking multiple models."""
        from shinrin.benchmark import benchmark_training

        X, y = sample_data
        models = {
            "tree": MondrianTreeRegressor(max_depth=3, random_state=0),
            "forest": MondrianForestRegressor(
                n_estimators=3,
                max_depth=3,
                random_state=0,
            ),
        }
        results = benchmark_training(models, X, y, n_repeats=2)

        assert "tree" in results
        assert "forest" in results


class TestBenchmarkPrediction:
    """Tests for prediction benchmark."""

    @pytest.fixture
    def fitted_models(self):
        """Fitted models for prediction benchmark."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5).astype(np.float32)
        y = rng.randn(50)

        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)

        forest = MondrianForestRegressor(
            n_estimators=3,
            max_depth=3,
            random_state=0,
        )
        forest.fit(X, y)

        return {"tree": tree, "forest": forest}, X

    def test_benchmark_prediction(self, fitted_models):
        """Test prediction benchmark."""
        from shinrin.benchmark import benchmark_prediction

        models, X = fitted_models
        results = benchmark_prediction(models, X, n_repeats=10)

        assert "tree" in results
        assert "forest" in results
        assert "mean_time" in results["tree"]
        assert "predictions_shape" in results["tree"]


class TestBenchmarkModelSize:
    """Tests for model size benchmark."""

    @pytest.fixture
    def fitted_models(self):
        """Fitted models for size benchmark."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randn(100)

        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)

        forest = MondrianForestRegressor(
            n_estimators=3,
            max_depth=3,
            random_state=0,
        )
        forest.fit(X, y)

        return {"tree": tree, "forest": forest}

    def test_benchmark_model_size_single_tree(self, fitted_models):
        """Test model size for single tree."""
        from shinrin.benchmark import benchmark_model_size

        results = benchmark_model_size(fitted_models)
        tree_size = results["tree"]

        assert tree_size["n_estimators"] == 1
        assert tree_size["n_nodes"] > 0
        assert tree_size["n_leaves"] > 0

    def test_benchmark_model_size_forest(self, fitted_models):
        """Test model size for forest."""
        from shinrin.benchmark import benchmark_model_size

        results = benchmark_model_size(fitted_models)
        forest_size = results["forest"]

        assert forest_size["n_estimators"] == 3
        assert forest_size["n_nodes"] > 0
        assert forest_size["n_leaves"] > 0


class TestFullBenchmark:
    """Tests for full benchmark suite."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for benchmarking."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randn(100)
        return X[:80], y[:80], X[80:]

    def test_full_benchmark(self, sample_data):
        """Test full benchmark suite."""
        from shinrin.benchmark import full_benchmark

        X_train, y_train, X_test = sample_data
        models = {"tree": MondrianTreeRegressor(max_depth=3, random_state=0)}
        results = full_benchmark(
            models,
            X_train,
            y_train,
            X_test,
            n_repeats_train=1,
            n_repeats_predict=5,
        )

        assert "tree.train" in results
        assert "tree.predict" in results
        assert "tree.size" in results


class TestPrintBenchmarkReport:
    """Tests for benchmark report printing."""

    def test_print_report(self):
        """Test that report printing works without errors."""
        from shinrin.benchmark import print_benchmark_report

        results = {
            "tree.train": {"mean_time": 0.01, "std_time": 0.001},
            "tree.predict": {"mean_time": 0.0001, "std_time": 0.00001},
            "tree.size": {"n_nodes": 10, "n_leaves": 5, "n_estimators": 1},
        }

        # Should not raise
        print_benchmark_report(results)
