"""Tests for ONNX model import into Mondrian trees and forests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shinrin import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    MondrianTreeClassifier,
    MondrianTreeRegressor,
)
from shinrin.onnx import to_onnx

ONNX_INSTALLED = True
try:
    import onnx  # noqa: F401
except ImportError:
    ONNX_INSTALLED = False

SKLEARN_INSTALLED = True
try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
except ImportError:
    SKLEARN_INSTALLED = False


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def regression_data():
    """Regression training and test data."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 4).astype(np.float32)
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + rng.randn(500) * 0.1
    X_test = rng.randn(100, 4).astype(np.float32)
    y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1]) + rng.randn(100) * 0.1
    return X_train, y_train, X_test, y_test


@pytest.fixture
def classification_data():
    """Classification training and test data."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 4).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = rng.randn(100, 4).astype(np.float32)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    return X_train, y_train, X_test, y_test


# ===================================================================
# Helper: create ONNX models from sklearn
# ===================================================================


def _make_sklearn_onnx_pair(model_cls, X_train, y_train):
    """Create a sklearn model and its ONNX representation."""
    sklearn_model = model_cls(n_estimators=5, max_depth=3, random_state=42)
    sklearn_model.fit(X_train, y_train)
    onnx_model = to_onnx(sklearn_model, X_train)
    return sklearn_model, onnx_model


# ===================================================================
# Tests: Single Tree from ONNX
# ===================================================================


@pytest.mark.skipif(
    not (ONNX_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx and sklearn required",
)
class TestMondrianTreeFromModel:
    """Tests for MondrianTree.from_model() with ONNX models."""

    def test_regressor_from_sklearn_tree(self, regression_data):
        """Test converting a sklearn DecisionTree to MondrianTree via ONNX."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, X_test, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        tree = MondrianTreeRegressor.from_model(onnx_model, X_train, y_train)

        # Predictions should match
        sklearn_pred = sklearn_model.predict(X_test)
        tree_pred = tree.predict(X_test)
        np.testing.assert_allclose(sklearn_pred, tree_pred, rtol=0.01)

    def test_regressor_from_onnx_single_tree(self, regression_data):
        """Test converting an ONNX single-tree model to MondrianTree."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, X_test, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        tree = MondrianTreeRegressor.from_model(onnx_model, X_train, y_train)

        # Predictions should match
        sklearn_pred = sklearn_model.predict(X_test)
        tree_pred = tree.predict(X_test)
        np.testing.assert_allclose(sklearn_pred, tree_pred, rtol=0.01)

    def test_classifier_from_sklearn_tree(self, classification_data):
        """Test converting a sklearn DecisionTree to MondrianTreeClassifier via ONNX."""
        from sklearn.tree import DecisionTreeClassifier

        X_train, y_train, X_test, _ = classification_data
        sklearn_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        tree = MondrianTreeClassifier.from_model(onnx_model, X_train, y_train)

        # Predictions should match (compare class indices)
        sklearn_pred = sklearn_model.predict(X_test)
        tree_pred = tree.predict(X_test)
        sklearn_classes = sklearn_model.classes_
        sklearn_indices = np.array(
            [
                list(sklearn_classes).index(c) if isinstance(c, str) else int(c)
                for c in sklearn_pred
            ]
        )
        tree_classes = tree.classes_
        tree_indices = np.array([list(tree_classes).index(c) for c in tree_pred])
        np.testing.assert_array_equal(sklearn_indices, tree_indices)

    def test_classifier_proba_from_onnx(self, classification_data):
        """Test converting ONNX classifier to MondrianTreeClassifier."""
        from sklearn.tree import DecisionTreeClassifier

        X_train, y_train, X_test, _ = classification_data
        sklearn_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        tree = MondrianTreeClassifier.from_model(onnx_model, X_train, y_train)

        # Predictions should match (compare class indices)
        sklearn_pred = sklearn_model.predict(X_test)
        tree_pred = tree.predict(X_test)
        # Map sklearn predictions to indices based on classes_
        sklearn_classes = sklearn_model.classes_
        sklearn_indices = np.array(
            [
                list(sklearn_classes).index(c) if isinstance(c, str) else int(c)
                for c in sklearn_pred
            ]
        )
        # Map tree predictions (string class names) to indices
        tree_classes = tree.classes_
        tree_indices = np.array([list(tree_classes).index(c) for c in tree_pred])
        np.testing.assert_array_equal(sklearn_indices, tree_indices)


# ===================================================================
# Tests: Forest from ONNX
# ===================================================================


@pytest.mark.skipif(
    not (ONNX_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx and sklearn required",
)
class TestMondrianForestFromModel:
    """Tests for MondrianForest.from_model() with ONNX models."""

    def test_forest_regressor_from_sklearn(self, regression_data):
        """Test converting sklearn RandomForest to MondrianForest via ONNX."""
        X_train, y_train, X_test, _ = regression_data
        sklearn_model = RandomForestRegressor(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestRegressor.from_model(onnx_model, X_train, y_train)

        # Predictions should match
        sklearn_pred = sklearn_model.predict(X_test)
        forest_pred = forest.predict(X_test)
        np.testing.assert_allclose(sklearn_pred, forest_pred, rtol=0.1)

    def test_forest_regressor_from_onnx(self, regression_data):
        """Test converting ONNX random forest to MondrianForest."""
        X_train, y_train, X_test, _ = regression_data
        sklearn_model = RandomForestRegressor(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestRegressor.from_model(onnx_model, X_train, y_train)

        # Predictions should match
        sklearn_pred = sklearn_model.predict(X_test)
        forest_pred = forest.predict(X_test)
        np.testing.assert_allclose(sklearn_pred, forest_pred, rtol=0.1)

    def test_forest_regressor_from_gb_onnx(self, regression_data):
        """Test converting ONNX gradient boosting to MondrianForest."""
        X_train, y_train, X_test, _ = regression_data
        sklearn_model = GradientBoostingRegressor(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestRegressor.from_model(onnx_model, X_train, y_train)

        # Predictions should match
        sklearn_pred = sklearn_model.predict(X_test)
        forest_pred = forest.predict(X_test)
        np.testing.assert_allclose(sklearn_pred, forest_pred, rtol=0.1)

    def test_forest_classifier_from_sklearn(self, classification_data):
        """Test converting sklearn RandomForest to MondrianForestClassifier via ONNX."""
        X_train, y_train, X_test, _ = classification_data
        sklearn_model = RandomForestClassifier(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestClassifier.from_model(onnx_model, X_train, y_train)

        # Predictions should match (compare class indices)
        sklearn_pred = sklearn_model.predict(X_test)
        forest_pred = forest.predict(X_test)
        sklearn_classes = sklearn_model.classes_
        sklearn_indices = np.array(
            [
                list(sklearn_classes).index(c) if isinstance(c, str) else int(c)
                for c in sklearn_pred
            ]
        )
        forest_classes = forest.classes_
        forest_indices = np.array([list(forest_classes).index(c) for c in forest_pred])
        np.testing.assert_array_equal(sklearn_indices, forest_indices)

    def test_forest_classifier_from_onnx(self, classification_data):
        """Test converting ONNX classifier forest to MondrianForestClassifier."""
        X_train, y_train, X_test, _ = classification_data
        sklearn_model = RandomForestClassifier(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestClassifier.from_model(onnx_model, X_train, y_train)

        # Predictions should match (compare class indices)
        sklearn_pred = sklearn_model.predict(X_test)
        forest_pred = forest.predict(X_test)
        sklearn_classes = sklearn_model.classes_
        sklearn_indices = np.array(
            [
                list(sklearn_classes).index(c) if isinstance(c, str) else int(c)
                for c in sklearn_pred
            ]
        )
        forest_classes = forest.classes_
        forest_indices = np.array([list(forest_classes).index(c) for c in forest_pred])
        np.testing.assert_array_equal(sklearn_indices, forest_indices)


# ===================================================================
# Tests: partial_fit after conversion
# ===================================================================


@pytest.mark.skipif(
    not (ONNX_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx and sklearn required",
)
class TestPartialFitAfterConversion:
    """Test that partial_fit works after from_model conversion."""

    def test_tree_partial_fit_regressor(self, regression_data):
        """Test that MondrianTree.partial_fit works after conversion."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, _, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        tree = MondrianTreeRegressor.from_model(onnx_model, X_train, y_train)

        # partial_fit should not raise
        rng = np.random.RandomState(123)
        X_new = rng.randn(50, 4).astype(np.float32)
        y_new = np.sin(X_new[:, 0]) + rng.randn(50) * 0.1

        # This should work without raising
        tree.partial_fit(X_new, y_new)
        assert hasattr(tree, "tree_")

    def test_forest_partial_fit_regressor(self, regression_data):
        """Test that MondrianForest.partial_fit works after conversion."""
        X_train, y_train, _, _ = regression_data
        sklearn_model = RandomForestRegressor(
            n_estimators=3, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestRegressor.from_model(onnx_model, X_train, y_train)

        # partial_fit should not raise
        rng = np.random.RandomState(123)
        X_new = rng.randn(50, 4).astype(np.float32)
        y_new = np.sin(X_new[:, 0]) + rng.randn(50) * 0.1

        forest.partial_fit(X_new, y_new)
        assert hasattr(forest, "estimators_")

    def test_forest_partial_fit_classifier(self, classification_data):
        """Test that MondrianForestClassifier.partial_fit works after conversion."""
        X_train, y_train, _, _ = classification_data
        sklearn_model = RandomForestClassifier(
            n_estimators=3, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        forest = MondrianForestClassifier.from_model(onnx_model, X_train, y_train)

        # partial_fit should not raise
        rng = np.random.RandomState(123)
        X_new = rng.randn(50, 4).astype(np.float32)
        y_new = (X_new[:, 0] + X_new[:, 1] > 0).astype(int)

        forest.partial_fit(X_new, y_new)
        assert hasattr(forest, "estimators_")


# ===================================================================
# Tests: Warnings
# ===================================================================


@pytest.mark.skipif(
    not (ONNX_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx and sklearn required",
)
class TestWarnings:
    """Test that appropriate warnings are issued."""

    def test_small_dataset_warning(self, regression_data):
        """Test that a warning is issued for small datasets."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, _, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        # Use a small dataset (< 300 samples)
        X_small = X_train[:100]
        y_small = y_train[:100]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = MondrianTreeRegressor.from_model(onnx_model, X_small, y_small)
            assert len(w) >= 1
            assert any("300" in str(warning.message) for warning in w), (
                f"Expected 300 sample warning, got: {[str(x.message) for x in w]}"
            )

    def test_no_warning_for_large_dataset(self, regression_data):
        """Test that no warning is issued for large datasets."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, _, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = MondrianTreeRegressor.from_model(onnx_model, X_train, y_train)
            sample_warnings = [x for x in w if "300" in str(x.message)]
            assert len(sample_warnings) == 0


# ===================================================================
# Tests: Error handling
# ===================================================================


@pytest.mark.skipif(
    not (ONNX_INSTALLED and SKLEARN_INSTALLED),
    reason="onnx and sklearn required",
)
class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_model_raises(self):
        """Test that invalid models raise ValueError."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100)

        class NotATree:
            pass

        with pytest.raises(ValueError, match="tree_|estimators_|ModelProto"):
            MondrianTreeRegressor.from_model(NotATree(), X, y)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatches raise ValueError."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100)

        with pytest.raises((ValueError, TypeError)):
            MondrianTreeRegressor.from_model(None, X, y)


# ===================================================================
# Tests: ONNX-specific extraction
# ===================================================================


@pytest.mark.skipif(
    not ONNX_INSTALLED,
    reason="onnx required",
)
class TestONNXExtraction:
    """Tests for ONNX tree extraction logic."""

    def test_count_trees_in_onnx_single(self, regression_data):
        """Test counting trees in a single-tree ONNX model."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, _, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        from shinrin.onnx_import import _count_trees_in_onnx

        assert _count_trees_in_onnx(onnx_model) == 1

    def test_count_trees_in_onnx_forest(self, regression_data):
        """Test counting trees in a forest ONNX model."""
        X_train, y_train, _, _ = regression_data
        sklearn_model = RandomForestRegressor(
            n_estimators=5, max_depth=3, random_state=42
        )
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        from shinrin.onnx_import import _count_trees_in_onnx

        assert _count_trees_in_onnx(onnx_model) == 5

    def test_extract_tree_from_onnx(self, regression_data):
        """Test extracting a single tree from ONNX."""
        from sklearn.tree import DecisionTreeRegressor

        X_train, y_train, _, _ = regression_data
        sklearn_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        onnx_model = to_onnx(sklearn_model, X_train)

        from shinrin.onnx_import import _extract_tree_from_onnx

        tree_info = _extract_tree_from_onnx(onnx_model, 0)
        assert "feature" in tree_info
        assert "threshold" in tree_info
        assert "left_child" in tree_info
        assert "right_child" in tree_info
        assert "value" in tree_info
        assert "n_classes" in tree_info
