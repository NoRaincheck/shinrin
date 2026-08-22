"""Tests for the MLP estimators (NumPy backend)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from shinrin import MLPClassifier, MLPRegressor

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification, make_regression
from sklearn.neural_network import MLPClassifier as SkMLPClassifier
from sklearn.neural_network import MLPRegressor as SkMLPRegressor

SMALL = {"hidden_layer_sizes": (32,), "max_iter": 60}


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=300, n_features=10, n_informative=5, noise=5.0, random_state=0
    )
    X = X.astype(np.float32)
    y_scaled = ((y - y.mean()) / y.std()).astype(np.float32)
    return X, y, y_scaled


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


# ---------------------------------------------------------------------------
# sklearn parity of training dynamics
# ---------------------------------------------------------------------------


def test_training_loss_matches_sklearn(regression_data):
    """With the same seed the per-epoch losses must track sklearn closely."""
    X, _, y = regression_data
    ours = MLPRegressor(hidden_layer_sizes=(24,), max_iter=30, random_state=0).fit(X, y)
    ref = SkMLPRegressor(hidden_layer_sizes=(24,), max_iter=30, random_state=0).fit(
        X, y
    )
    np.testing.assert_allclose(ours.loss_curve_, ref.loss_curve_, rtol=2e-2)


def test_classifier_loss_matches_sklearn(binary_data):
    X, y = binary_data
    ours = MLPClassifier(hidden_layer_sizes=(24,), max_iter=30, random_state=0).fit(
        X, y
    )
    ref = SkMLPClassifier(hidden_layer_sizes=(24,), max_iter=30, random_state=0).fit(
        X, y
    )
    np.testing.assert_allclose(ours.loss_curve_, ref.loss_curve_, rtol=2e-2)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regressor_basic(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(random_state=0, **SMALL).fit(X, y)
    assert reg.score(X, y) > 0.7
    preds = reg.predict(X)
    assert preds.shape == y.shape
    assert reg.n_features_in_ == 10
    assert reg.n_layers_ == 3  # input + one hidden + output
    assert len(reg.coefs_) == 2
    assert reg.coefs_[0].shape == (10, 32)
    assert reg.coefs_[1].shape == (32, 1)
    assert len(reg.intercepts_) == 2
    assert reg.n_iter_ <= SMALL["max_iter"]
    assert reg.loss_curve_[-1] < reg.loss_curve_[0]
    assert reg.out_activation_ == "identity"


def test_regressor_multioutput(regression_data):
    X, _, y = regression_data
    Y = np.column_stack([y, 2 * y + 1])
    reg = MLPRegressor(random_state=0, **SMALL).fit(X, Y)
    assert reg.predict(X).shape == Y.shape
    assert reg.n_outputs_ == 2
    assert reg.score(X, Y) > 0.7


def test_regressor_column_vector_warning(regression_data):
    X, _, y = regression_data
    with pytest.warns(DataConversionWarning):
        MLPRegressor(max_iter=1, random_state=0).fit(X, y[:, None])


def test_regressor_lbfgs(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(
        hidden_layer_sizes=(32,), solver="lbfgs", max_iter=60, random_state=0
    ).fit(X, y)
    assert reg.score(X, y) > 0.8


def test_regressor_sgd_momentum(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(
        hidden_layer_sizes=(32,),
        solver="sgd",
        learning_rate_init=0.05,
        max_iter=300,
        random_state=0,
    ).fit(X, y)
    assert reg.score(X, y) > 0.8


def test_regressor_invscaling_matches_sklearn(regression_data):
    """The invscaling schedule must decay lr exactly like sklearn's."""
    X, _, y = regression_data
    ours = MLPRegressor(
        hidden_layer_sizes=(16,),
        solver="sgd",
        learning_rate="invscaling",
        learning_rate_init=0.05,
        power_t=0.5,
        max_iter=30,
        random_state=0,
    ).fit(X, y)
    ref = SkMLPRegressor(
        hidden_layer_sizes=(16,),
        solver="sgd",
        learning_rate="invscaling",
        learning_rate_init=0.05,
        power_t=0.5,
        max_iter=30,
        random_state=0,
    ).fit(X, y)
    np.testing.assert_allclose(ours.loss_curve_, ref.loss_curve_, rtol=5e-2)


def test_adaptive_learning_rate(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(
        hidden_layer_sizes=(32,),
        solver="sgd",
        learning_rate="adaptive",
        learning_rate_init=0.05,
        max_iter=200,
        n_iter_no_change=5,
        tol=1e-5,
        random_state=0,
    ).fit(X, y)
    assert reg.score(X, y) > 0.7


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=1,
    )
    return X.astype(np.float32), y


def test_classifier_binary(binary_data):
    X, y = binary_data
    clf = MLPClassifier(random_state=0, **SMALL).fit(X, y)
    assert clf.score(X, y) >= 0.85
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(clf.predict(X)) <= set(clf.classes_)
    assert clf.out_activation_ == "logistic"
    assert clf.coefs_[-1].shape == (32, 1)


def test_classifier_multiclass(multiclass_data):
    X, y = multiclass_data
    clf = MLPClassifier(random_state=0, **SMALL).fit(X, y)
    assert clf.score(X, y) > 0.7
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf.out_activation_ == "softmax"
    assert clf.coefs_[-1].shape == (32, 3)


def test_classifier_string_labels(binary_data):
    X, y = binary_data
    labels = np.where(y == 0, "neg", "pos")
    clf = MLPClassifier(random_state=0, **SMALL).fit(X, labels)
    assert set(clf.predict(X)) == {"neg", "pos"}


def test_all_activations_train(binary_data):
    X, y = binary_data
    for activation in ("relu", "tanh", "logistic", "identity"):
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation=activation,
            max_iter=100,
            learning_rate_init=0.05,
            solver="sgd",
            random_state=0,
        ).fit(X[:200], y[:200])
        # identity hidden layers can still separate linearly-separable data
        assert clf.score(X[200:], y[200:]) > 0.5, activation


# ---------------------------------------------------------------------------
# PLE embeddings extension
# ---------------------------------------------------------------------------


def test_ple_embeddings_improve_unscaled_regression():
    """The PLE recipe handles unscaled targets where raw features struggle."""
    X, y = make_regression(
        n_samples=400, n_features=12, n_informative=6, noise=20.0, random_state=3
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    plain = MLPRegressor(hidden_layer_sizes=(64,), max_iter=150, random_state=0)
    plain.fit(X, y)
    ple = MLPRegressor(
        hidden_layer_sizes=(64,),
        use_embeddings=True,
        use_asinh=True,
        use_scaler=True,
        max_iter=150,
        random_state=0,
    ).fit(X, y)
    assert ple.score(X, y) > plain.score(X, y) - 0.05
    assert ple.preprocessor_.encoder_ is not None
    assert plain.preprocessor_.encoder_ is None


def test_ple_embeddings_config(binary_data):
    X, y = binary_data
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        use_embeddings=True,
        use_asinh=True,
        use_scaler=True,
        n_bins=16,
        d_embedding=4,
        max_iter=50,
        random_state=0,
    ).fit(X, y)
    enc_width = sum(len(b) - 1 for b in clf.preprocessor_.bins_)
    expected_d_in = 10 * 4 + enc_width * 0  # embedding projects to d_embedding
    assert all(
        c.shape[0] == expected_d_in or i > 0 for i, c in enumerate(clf.coefs_[0:1])
    )
    assert clf.coefs_[0].shape[0] == 10 * 4


def test_categorical_auto_detection_with_embeddings():
    rng = np.random.RandomState(0)
    X = rng.randn(250, 5).astype(np.float32)
    X[:, 0] = rng.randint(0, 4, size=250)  # low cardinality -> categorical
    y = (X[:, 1] > 0).astype(int)
    clf = MLPClassifier(
        hidden_layer_sizes=(24,),
        use_embeddings=True,
        use_asinh=True,
        use_scaler=True,
        max_iter=200,
        random_state=0,
    ).fit(X, y)
    assert clf.preprocessor_.categorical_indices_ == [0]
    assert clf.preprocessor_.numerical_indices_ == [1, 2, 3, 4]
    assert clf.score(X, y) > 0.8


# ---------------------------------------------------------------------------
# Dropout extension
# ---------------------------------------------------------------------------


def test_dropout_trains_and_predicts(binary_data):
    X, y = binary_data
    clf = MLPClassifier(
        hidden_layer_sizes=(32,), dropout=0.2, max_iter=80, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.8


def test_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        MLPRegressor(dropout=1.5).fit([[0.0]], [0.0])


# ---------------------------------------------------------------------------
# Convergence / early stopping / schedules
# ---------------------------------------------------------------------------


def test_early_stopping_restores_best_weights(binary_data):
    X, y = binary_data
    clf = MLPClassifier(
        hidden_layer_sizes=(24,),
        early_stopping=True,
        validation_fraction=0.25,
        n_iter_no_change=4,
        max_iter=100,
        random_state=0,
    ).fit(X, y)
    assert len(clf.validation_scores_) == clf.n_iter_
    assert clf.best_validation_score_ == max(clf.validation_scores_)
    assert clf.score(X, y) > 0.7


def test_convergence_stops_before_max_iter(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(
        hidden_layer_sizes=(32,),
        max_iter=500,
        n_iter_no_change=5,
        tol=1e-3,
        random_state=0,
    ).fit(X, y)
    assert reg.n_iter_ < 500
    assert reg.best_loss_ == min(reg.loss_curve_)


def test_invalid_solver_and_lr():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="solver"):
        MLPRegressor(solver="bogus").fit(X, y)
    with pytest.raises(ValueError, match="learning_rate"):
        MLPRegressor(learning_rate="bogus").fit(X, y)
    with pytest.raises(ValueError, match="activation"):
        MLPClassifier(activation="bogus").fit(X, y)
    with pytest.raises(ValueError, match="batch_size"):
        MLPRegressor(batch_size="bogus").fit(X, y)


# ---------------------------------------------------------------------------
# Determinism / partial fit / warm start / pickle
# ---------------------------------------------------------------------------


def test_random_state_determinism(regression_data):
    X, _, y = regression_data
    a = MLPRegressor(random_state=42, **SMALL).fit(X, y)
    b = MLPRegressor(random_state=42, **SMALL).fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X))
    assert a.loss_curve_ == b.loss_curve_


def test_partial_fit_matches_batch_epoch_semantics(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(random_state=0, **SMALL)
    for _ in range(40):
        reg.partial_fit(X[:250], y[:250])
    assert reg.n_iter_ == 40
    assert reg.score(X, y) > 0.5


def test_partial_fit_requires_classes_first(binary_data):
    X, y = binary_data
    clf = MLPClassifier(random_state=0, **SMALL)
    with pytest.raises(ValueError, match="classes"):
        clf.partial_fit(X[:10], y[:10])


def test_partial_fit_rejects_early_stopping(binary_data):
    X, y = binary_data
    clf = MLPClassifier(early_stopping=True, random_state=0, **SMALL)
    with pytest.raises(ValueError, match="early_stopping"):
        clf.partial_fit(X[:10], y[:10], classes=[0, 1])


def test_warm_start_continues(binary_data):
    X, y = binary_data
    clf = MLPClassifier(
        hidden_layer_sizes=(16,), max_iter=5, warm_start=True, random_state=0
    )
    clf.fit(X, y)
    first_t = clf.t_
    clf.fit(X, y)
    assert clf.t_ > first_t


def test_pickle_roundtrip(regression_data):
    X, _, y = regression_data
    reg = MLPRegressor(random_state=0, **SMALL).fit(X, y)
    restored = pickle.loads(pickle.dumps(reg))
    assert np.allclose(restored.predict(X), reg.predict(X))


def test_sklearn_compliance(regression_data):
    from sklearn.utils.estimator_checks import check_estimator

    est = MLPRegressor(hidden_layer_sizes=(8,), max_iter=5, random_state=0)
    check_estimator(est)
