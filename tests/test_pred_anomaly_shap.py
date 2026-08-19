"""Tests for pred_anomaly and pred_contribs methods."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from shinrin import MondrianForestRegressor, MondrianForestClassifier
from shinrin._skgarden.mondrian.tree import MondrianTreeRegressor, MondrianTreeClassifier


class TestTreePredAnomaly:
    """Test pred_anomaly on single Mondrian trees."""

    @pytest.fixture
    def X_train(self):
        return np.random.randn(100, 4).astype(np.float32)

    @pytest.fixture
    def y_train_reg(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def X_test(self):
        return np.random.randn(10, 4).astype(np.float32)

    def test_tree_anomaly_returns_array(self, X_train, y_train_reg, X_test):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        anomaly = tree._compute_anomaly(X_test)
        assert isinstance(anomaly, np.ndarray)
        assert anomaly.shape == (10,)

    def test_tree_anomaly_predict(self, X_train, y_train_reg, X_test):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        pred, anomaly = tree.predict(X_test, return_anomaly=True)
        assert anomaly.shape == (10,)
        assert pred.shape == (10,)


class TestTreePredContribs:
    """Test pred_contribs on single Mondrian trees."""

    @pytest.fixture
    def X_train(self):
        return np.random.randn(100, 4).astype(np.float32)

    @pytest.fixture
    def y_train_reg(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def y_train_clf(self):
        return np.random.choice([0, 1, 2], size=100)

    @pytest.fixture
    def X_test(self):
        return np.random.randn(10, 4).astype(np.float32)

    def test_tree_shap_shape_regression(self, X_train, y_train_reg, X_test):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        pred, shap = tree.predict(X_test, return_shap=True)
        assert shap.shape == (10, 4)

    def test_tree_pred_contribs_shape_regression(
        self, X_train, y_train_reg, X_test
    ):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        contribs = tree.pred_contribs(X_test)
        assert contribs.shape == (10, 5)  # 4 features + 1 base value

    def test_tree_shap_additive_regression(
        self, X_train, y_train_reg, X_test
    ):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        pred = tree.predict(X_test)
        contribs = tree.pred_contribs(X_test)
        pred_from_shap = contribs[:, :-1].sum(axis=1) + contribs[:, -1]
        assert np.allclose(pred, pred_from_shap, atol=1e-3)

    def test_tree_classifier_return_std_raises(self, X_train, y_train_clf, X_test):
        tree = MondrianTreeClassifier(random_state=42)
        tree.fit(X_train, y_train_clf)
        with pytest.raises(ValueError, match="return_std is not supported"):
            tree.predict(X_test, return_std=True)

    def test_tree_shap_shape_classification(
        self, X_train, y_train_clf, X_test
    ):
        tree = MondrianTreeClassifier(random_state=42)
        tree.fit(X_train, y_train_clf)
        pred, shap = tree.predict(X_test, return_shap=True)
        # Classification returns per-class SHAP: (n_samples, n_features, n_classes)
        assert shap.shape == (10, 4, 3)

    def test_tree_pred_contribs_shape_classification(
        self, X_train, y_train_clf, X_test
    ):
        tree = MondrianTreeClassifier(random_state=42)
        tree.fit(X_train, y_train_clf)
        contribs = tree.pred_contribs(X_test)
        assert contribs.ndim == 3
        assert contribs.shape == (10, 5, 3)  # 4 features + 1 base, 3 classes

    def test_tree_predict_all_kwargs(
        self, X_train, y_train_reg, X_test
    ):
        tree = MondrianTreeRegressor(random_state=42)
        tree.fit(X_train, y_train_reg)
        result = tree.predict(
            X_test,
            return_std=True,
            return_anomaly=True,
            return_shap=True,
        )
        assert len(result) == 4
        pred, std, anomaly, shap = result
        assert pred.shape == (10,)
        assert std.shape == (10,)
        assert anomaly.shape == (10,)
        assert shap.shape == (10, 4)


class TestForestPredAnomaly:
    """Test pred_anomaly on Mondrian forests."""

    @pytest.fixture
    def X_train(self):
        return np.random.randn(100, 4).astype(np.float32)

    @pytest.fixture
    def y_train_reg(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def y_train_clf(self):
        return np.random.choice([0, 1, 2], size=100)

    @pytest.fixture
    def X_test(self):
        return np.random.randn(10, 4).astype(np.float32)

    def test_forest_anomaly_shape_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        anomaly = forest.pred_anomaly(X_test)
        assert anomaly.shape == (10,)

    def test_forest_anomaly_range_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        anomaly = forest.pred_anomaly(X_test)
        assert anomaly.min() >= 0.0
        assert anomaly.max() <= 1.0

    def test_forest_anomaly_consistent(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        anomaly_direct = forest.pred_anomaly(X_test)
        _, anomaly_from_pred = forest.predict(X_test, return_anomaly=True)
        assert np.allclose(anomaly_direct, anomaly_from_pred)

    def test_forest_anomaly_classifier(
        self, X_train, y_train_clf, X_test
    ):
        forest = MondrianForestClassifier(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_clf)
        anomaly = forest.pred_anomaly(X_test)
        assert anomaly.shape == (10,)
        assert anomaly.min() >= 0.0
        assert anomaly.max() <= 1.0


class TestForestPredContribs:
    """Test pred_contribs on Mondrian forests."""

    @pytest.fixture
    def X_train(self):
        return np.random.randn(100, 4).astype(np.float32)

    @pytest.fixture
    def y_train_reg(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def y_train_clf(self):
        return np.random.choice([0, 1, 2], size=100)

    @pytest.fixture
    def X_test(self):
        return np.random.randn(10, 4).astype(np.float32)

    def test_forest_shap_shape_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        pred, shap = forest.predict(X_test, return_shap=True)
        assert shap.shape == (10, 4)

    def test_forest_pred_contribs_shape_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        contribs = forest.pred_contribs(X_test)
        assert contribs.shape == (10, 5)

    def test_forest_shap_additive_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        pred = forest.predict(X_test)
        contribs = forest.pred_contribs(X_test)
        pred_from_shap = contribs[:, :-1].sum(axis=1) + contribs[:, -1]
        assert np.allclose(pred, pred_from_shap, atol=1e-2)

    def test_forest_pred_contribs_shape_classification(
        self, X_train, y_train_clf, X_test
    ):
        forest = MondrianForestClassifier(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_clf)
        pred, shap = forest.predict(X_test, return_shap=True)
        # Classification returns per-class SHAP: (n_samples, n_features, n_classes)
        assert shap.shape == (10, 4, 3)
        contribs = forest.pred_contribs(X_test)
        # pred_contribs returns (n_samples, n_features + 1, n_classes)
        assert contribs.ndim == 3
        assert contribs.shape == (10, 5, 3)


class TestPredictWithMultipleKwargs:
    """Test predict() with multiple return kwargs combined."""

    @pytest.fixture
    def X_train(self):
        return np.random.randn(100, 4).astype(np.float32)

    @pytest.fixture
    def y_train_reg(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def X_test(self):
        return np.random.randn(10, 4).astype(np.float32)

    def test_forest_predict_all_kwargs_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        result = forest.predict(
            X_test,
            return_std=True,
            return_anomaly=True,
            return_shap=True,
        )
        assert len(result) == 4
        pred, std, anomaly, shap = result
        assert pred.shape == (10,)
        assert std.shape == (10,)
        assert anomaly.shape == (10,)
        assert shap.shape == (10, 4)

    def test_forest_predict_anomaly_and_shap_regression(
        self, X_train, y_train_reg, X_test
    ):
        forest = MondrianForestRegressor(n_estimators=10, random_state=42)
        forest.fit(X_train, y_train_reg)
        pred, anomaly, shap = forest.predict(
            X_test, return_anomaly=True, return_shap=True
        )
        assert pred.shape == (10,)
        assert anomaly.shape == (10,)
        assert shap.shape == (10, 4)
