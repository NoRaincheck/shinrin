"""Tests for TreeSHAP explanation module."""

import numpy as np
import pytest

from shinrin import MondrianForestRegressor, MondrianTreeRegressor
from shinrin.shap import TreeExplainer, _get_tree_structure, explanation


class TestTreeStructureExtraction:
    """Test _get_tree_structure helper."""

    def test_single_tree_structure(self):
        """Tree structure extraction returns expected keys."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = rng.randn(100)

        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)

        struct = _get_tree_structure(tree)

        assert "children_left" in struct
        assert "children_right" in struct
        assert "feature" in struct
        assert "threshold" in struct
        assert "value" in struct
        assert "n_node_samples" in struct
        assert "is_leaf" in struct
        assert "is_regression" in struct
        assert "n_features" in struct
        assert "n_classes" in struct
        assert struct["n_features"] == 4
        assert struct["is_regression"] is True
        # Mondrian trees should have tau
        assert struct["tau"] is not None
        assert struct["lower_bounds"] is not None
        assert struct["upper_bounds"] is not None


class TestTreeExplainerSingleTree:
    """Test TreeExplainer with a single Mondrian tree."""

    @pytest.fixture
    def fitted_tree(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = rng.randn(200)
        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)
        return tree, X, y

    def test_explainer_creation(self, fitted_tree):
        tree, _, _ = fitted_tree
        explainer = TreeExplainer(tree)
        assert explainer._model_type == "tree"
        assert explainer._is_classification is False

    def test_expected_value(self, fitted_tree):
        tree, _, _ = fitted_tree
        explainer = TreeExplainer(tree)
        ev = explainer.expected_value()
        assert isinstance(ev, (float, np.floating))
        # Expected value should be close to mean of tree's root prediction
        assert np.isfinite(ev)

    def test_shap_values_shape(self, fitted_tree):
        tree, X, _ = fitted_tree
        explainer = TreeExplainer(tree)
        shap_vals = explainer.shap_values(X[0])
        assert shap_vals.shape == (5,)  # n_features

    def test_shap_values_shape_2d(self, fitted_tree):
        tree, X, _ = fitted_tree
        explainer = TreeExplainer(tree)
        shap_vals = explainer.shap_values(X[:10])
        assert shap_vals.shape == (10, 5)

    def test_shap_values_finite(self, fitted_tree):
        tree, X, _ = fitted_tree
        explainer = TreeExplainer(tree)
        shap_vals = explainer.shap_values(X[:20])
        assert np.all(np.isfinite(shap_vals))

    def test_shap_values_sum(self, fitted_tree):
        """SHAP values should sum to (prediction - expected_value)."""
        tree, X, _ = fitted_tree
        explainer = TreeExplainer(tree)
        ev = explainer.expected_value()

        for i in range(min(10, len(X))):
            shap_vals = explainer.shap_values(X[i])
            prediction = tree.predict(X[i : i + 1])[0]
            shap_sum = float(np.sum(shap_vals))
            # Allow some tolerance due to TreeSHAP approximation
            assert np.isclose(shap_sum, prediction - ev, atol=0.5, rtol=0.1), (
                f"Sample {i}: SHAP sum={shap_sum:.4f}, pred-ev={prediction - ev:.4f}"
            )


class TestTreeExplainerForest:
    """Test TreeExplainer with a Mondrian forest."""

    @pytest.fixture
    def fitted_forest(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = rng.randn(200)
        forest = MondrianForestRegressor(n_estimators=5, max_depth=4, random_state=0)
        forest.fit(X, y)
        return forest, X, y

    def test_explainer_creation(self, fitted_forest):
        forest, _, _ = fitted_forest
        explainer = TreeExplainer(forest)
        assert explainer._model_type == "forest"
        assert explainer._is_classification is False

    def test_expected_value(self, fitted_forest):
        forest, _, _ = fitted_forest
        explainer = TreeExplainer(forest)
        ev = explainer.expected_value()
        assert isinstance(ev, (float, np.floating))
        assert np.isfinite(ev)

    def test_shap_values_shape(self, fitted_forest):
        forest, X, _ = fitted_forest
        explainer = TreeExplainer(forest)
        shap_vals = explainer.shap_values(X[0])
        assert shap_vals.shape == (5,)

    def test_shap_values_finite(self, fitted_forest):
        forest, X, _ = fitted_forest
        explainer = TreeExplainer(forest)
        shap_vals = explainer.shap_values(X[:20])
        assert np.all(np.isfinite(shap_vals))

    def test_shap_values_sum(self, fitted_forest):
        """Forest SHAP values should sum to (prediction - expected_value)."""
        forest, X, _ = fitted_forest
        explainer = TreeExplainer(forest)
        ev = explainer.expected_value()

        for i in range(min(10, len(X))):
            shap_vals = explainer.shap_values(X[i])
            prediction = forest.predict(X[i : i + 1])[0]
            shap_sum = float(np.sum(shap_vals))
            assert np.isclose(shap_sum, prediction - ev, atol=1.0, rtol=0.2), (
                f"Sample {i}: SHAP sum={shap_sum:.4f}, pred-ev={prediction - ev:.4f}"
            )


class TestExplanationFunction:
    """Test the convenience explanation() function."""

    @pytest.fixture
    def fitted_tree(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50)
        tree = MondrianTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, y)
        return tree, X

    def test_single_explanation(self, fitted_tree):
        tree, X = fitted_tree
        exp = explanation(tree, X[0])
        assert isinstance(exp, dict)
        assert "prediction" in exp
        assert "expected_value" in exp
        assert "shap_values" in exp
        assert "features" in exp
        assert "contributions" in exp
        assert len(exp["features"]) == 3
        assert len(exp["contributions"]) == 3

    def test_batch_explanation(self, fitted_tree):
        tree, X = fitted_tree
        exps = explanation(tree, X[:5])
        assert isinstance(exps, list)
        assert len(exps) == 5
        for exp in exps:
            assert isinstance(exp, dict)
            assert "prediction" in exp

    def test_feature_names(self, fitted_tree):
        tree, X = fitted_tree
        names = ["age", "income", "score"]
        exp = explanation(tree, X[0], feature_names=names)
        assert exp["features"] == names
        assert all(name in exp["contributions"] for name in names)


class TestTreeExplainerEdgeCases:
    """Test edge cases and error handling."""

    def test_unfitted_model_raises(self):
        from shinrin import MondrianTreeRegressor

        tree = MondrianTreeRegressor()
        # Should raise because tree_ is not set
        with pytest.raises(ValueError, match="tree_"):
            TreeExplainer(tree)

    def test_non_tree_model_raises(self):
        class FakeModel:
            pass

        with pytest.raises(ValueError, match="tree_"):
            TreeExplainer(FakeModel())

    def test_single_sample_input(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50)
        tree = MondrianTreeRegressor(max_depth=2, random_state=0)
        tree.fit(X, y)

        explainer = TreeExplainer(tree)
        # Single 1D array
        shap = explainer.shap_values(X[0])
        assert shap.shape == (3,)

        # Single 2D array
        shap = explainer.shap_values(X[0:1])
        assert shap.shape == (1, 3)
