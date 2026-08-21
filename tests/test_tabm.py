"""Tests for the vendored TabM estimators (NumPy backend)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from shinrin import TabMClassifier, TabMRegressor

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification, make_regression

SMALL = {"hidden_layer_sizes": (32,), "k": 4, "n_bins": 16}
SMALL_REG = {**SMALL, "max_iter": 150}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=300, n_features=10, n_informative=5, noise=5.0, random_state=0
    )
    return X.astype(np.float32), y.astype(np.float32)


def test_regressor_basic(regression_data):
    X, y = regression_data
    reg = TabMRegressor(random_state=0, **SMALL_REG)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.7
    preds = reg.predict(X)
    assert preds.shape == y.shape
    assert reg.n_features_in_ == 10
    assert reg.n_iter_ == SMALL_REG["max_iter"]
    assert len(reg.loss_curve_) == reg.n_iter_
    assert reg.loss_curve_[-1] < reg.loss_curve_[0]


def test_regressor_multioutput(regression_data):
    X, y = regression_data
    Y = np.column_stack([y, 2 * y + 1])
    reg = TabMRegressor(random_state=0, **SMALL_REG)
    reg.fit(X, Y)
    assert reg.predict(X).shape == Y.shape
    assert reg.score(X, Y) > 0.7


def test_regressor_lbfgs(regression_data):
    X, y = regression_data
    reg = TabMRegressor(
        hidden_layer_sizes=(32,), k=4, solver="lbfgs", max_iter=60, random_state=0
    )
    reg.fit(X, y)
    assert reg.score(X, y) > 0.8


def test_regressor_sgd(regression_data):
    X, y = regression_data
    reg = TabMRegressor(
        hidden_layer_sizes=(32,),
        k=4,
        solver="sgd",
        learning_rate_init=0.01,
        max_iter=300,
        random_state=0,
    )
    # Vanilla SGD needs small targets; unscaled y diverges (as with
    # sklearn's MLPRegressor).
    reg.fit(X, y / 100.0)
    assert reg.score(X, y / 100.0) > 0.5


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=0,
    )
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        random_state=1,
    )
    return X.astype(np.float32), y


def test_classifier_binary(binary_data):
    X, y = binary_data
    clf = TabMClassifier(random_state=0, **SMALL)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.85
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(clf.predict(X)) <= set(clf.classes_)
    assert clf.out_activation_ == "logistic"


def test_classifier_multiclass(multiclass_data):
    X, y = multiclass_data
    clf = TabMClassifier(random_state=0, **SMALL)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.7
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf.out_activation_ == "softmax"


def test_classifier_string_labels(binary_data):
    X, y = binary_data
    labels = np.where(y == 0, "neg", "pos")
    clf = TabMClassifier(random_state=0, **SMALL)
    clf.fit(X, labels)
    assert set(clf.predict(X)) == {"neg", "pos"}


# ---------------------------------------------------------------------------
# Preprocessing / categorical features
# ---------------------------------------------------------------------------


def test_categorical_auto_detection():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4).astype(np.float32)
    X[:, 0] = rng.randint(0, 3, size=200)  # low cardinality -> categorical
    y = (X[:, 1] > 0).astype(int)
    clf = TabMClassifier(random_state=0, **SMALL)
    clf.fit(X, y)
    assert clf.preprocessor_.categorical_indices_ == [0]
    assert clf.preprocessor_.numerical_indices_ == [1, 2, 3]
    assert clf.score(X, y) > 0.8


def test_categorical_explicit_indices():
    rng = np.random.RandomState(0)
    X = rng.randn(150, 3).astype(np.float32)
    X[:, 2] = rng.randint(0, 100, size=150)  # high cardinality but forced
    y = (X[:, 0] > 0).astype(int)
    clf = TabMClassifier(
        categorical_indices=[2],
        categorical_cardinality_threshold=0,
        random_state=0,
        **{k: v for k, v in SMALL.items() if k != "n_bins"},
    )
    clf.fit(X, y)
    assert clf.preprocessor_.categorical_indices_ == [2]


def test_embeddings_on_off(regression_data):
    X, y = regression_data
    for use_embeddings in (True, False):
        reg = TabMRegressor(
            use_embeddings=use_embeddings, random_state=0, max_iter=300, **SMALL
        )
        reg.fit(X, y)
        assert reg.score(X, y) > 0.6


def test_preprocessing_transforms_toggle(regression_data):
    X, y = regression_data
    reg = TabMRegressor(
        use_quantile=False,
        use_asinh=False,
        use_scaler=False,
        random_state=0,
        **SMALL,
    )
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


# ---------------------------------------------------------------------------
# Architecture variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini", "tabm-packed"])
def test_arch_types(regression_data, arch_type):
    X, y = regression_data
    reg = TabMRegressor(arch_type=arch_type, random_state=0, **SMALL)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


def test_invalid_arch_type():
    with pytest.raises(ValueError, match="arch_type"):
        TabMRegressor(arch_type="bogus").fit([[0.0]], [0.0])


def test_invalid_solver(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="solver"):
        TabMRegressor(solver="bogus").fit(X, y)


def test_invalid_activation(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="activation"):
        TabMRegressor(activation="tanh").fit(X, y)


# ---------------------------------------------------------------------------
# Determinism / early stopping / partial fit / pickle
# ---------------------------------------------------------------------------


def test_random_state_determinism(regression_data):
    X, y = regression_data
    a = TabMRegressor(random_state=42, **SMALL).fit(X, y)
    b = TabMRegressor(random_state=42, **SMALL).fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X))


def test_early_stopping(regression_data):
    X, y = regression_data
    reg = TabMRegressor(
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=3,
        random_state=0,
        **SMALL,
    )
    reg.fit(X, y)
    assert len(reg.validation_scores_) == reg.n_iter_
    assert reg.best_validation_score_ == min(reg.validation_scores_)
    assert reg.score(X, y) > 0.5


def test_partial_fit_regressor(regression_data):
    X, y = regression_data
    reg = TabMRegressor(random_state=0, **SMALL)
    for _ in range(60):
        reg.partial_fit(X[:250], y[:250] / 100.0)
    assert reg.n_iter_ == 60
    assert reg.score(X[250:], y[250:] / 100.0) > 0.5


def test_partial_fit_classifier(binary_data):
    X, y = binary_data
    clf = TabMClassifier(random_state=0, **SMALL)
    clf.partial_fit(X[:200], y[:200], classes=[0, 1])
    for _ in range(19):
        clf.partial_fit(X[200:], y[200:])
    assert clf.n_iter_ == 20
    assert clf.score(X, y) > 0.7


def test_partial_fit_requires_classes_first(binary_data):
    X, y = binary_data
    clf = TabMClassifier(random_state=0, **SMALL)
    with pytest.raises(ValueError, match="classes"):
        clf.partial_fit(X[:10], y[:10])


def test_pickle_roundtrip(regression_data):
    X, y = regression_data
    reg = TabMRegressor(random_state=0, **SMALL)
    reg.fit(X, y)
    restored = pickle.loads(pickle.dumps(reg))
    assert np.allclose(restored.predict(X), reg.predict(X))


def test_sklearn_compliance(regression_data):
    from sklearn.utils.estimator_checks import check_estimator

    # Run a subset of checks that apply to a non-standard deep model.
    est = TabMRegressor(hidden_layer_sizes=(8,), k=2, max_iter=5, random_state=0)
    check_estimator(est)
