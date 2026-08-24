"""Tests for the vendored GOSDT classifier (reference-ensemble optimal
sparse decision trees).

Ported/adapted from gosdt-guesses' test suite to the shinrin namespace.
Requires the ``sklearn`` optional extra.
"""

import numpy as np
import pytest

from shinrin import GOSDTClassifier, ThresholdGuessBinarizer
from shinrin._gosdt import Status

try:
    from sklearn.datasets import load_iris, make_classification
    from sklearn.ensemble import RandomForestClassifier
except ImportError:  # pragma: no cover - sklearn optional
    pytest.skip("scikit-learn not available", allow_module_level=True)


def test_single_class():
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_classes=1
    )
    clf = GOSDTClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(set(y_pred)) == 1


def test_toy_exact():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    clf = GOSDTClassifier(regularization=0.05, allow_small_reg=True)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(np.asarray(preds), y)

    result = clf.get_result()
    assert result["status"] == Status.CONVERGED
    assert result["model_loss"] == pytest.approx(0.0)


def test_iris_end_to_end():
    X, y = load_iris(return_X_y=True)
    enc = ThresholdGuessBinarizer(n_estimators=10, max_depth=2, random_state=0)
    X_bin = enc.fit_transform(X, y)

    clf = GOSDTClassifier(regularization=0.1, depth_budget=3, verbose=False)
    clf.fit(X_bin, y)

    accuracy = clf.score(X_bin, y)
    assert accuracy > 0.9

    proba = clf.predict_proba(X_bin)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    result = clf.get_result()
    assert result["status"] == Status.CONVERGED
    assert result["lower_bound"] <= result["upper_bound"] + 1e-6


def test_reference_ensemble_path():
    """The paper's core contribution: blackbox guesses guide the search."""
    X, y = make_classification(
        n_samples=300, n_features=12, n_informative=6, n_redundant=2, random_state=7
    )
    enc = ThresholdGuessBinarizer(n_estimators=10, max_depth=2, random_state=0)
    X_bin = enc.fit_transform(X, y)

    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_bin, y)
    y_ref = rf.predict(X_bin)

    clf_with_ref = GOSDTClassifier(regularization=0.05)
    clf_with_ref.fit(X_bin, y, y_ref=y_ref)

    clf_plain = GOSDTClassifier(regularization=0.05)
    clf_plain.fit(X_bin, y)

    assert clf_with_ref.result_.status == Status.CONVERGED
    assert clf_with_ref.score(X_bin, y) >= 0.7
    # Reference-guided search must find a model at least as good as plain.
    assert clf_with_ref.result_.model_loss <= clf_plain.result_.model_loss + 1e-6


def test_worker_limit_parity():
    """Parallel workers must return the same certified optimum as one worker.

    worker_limit > 1 exercises the engine's multi-threaded search path
    (worker_limit = 0 resolves to one worker per core inside the bridge).
    The optimal objective and its certified bounds are deterministic
    regardless of scheduling; individual trees may differ only among ties,
    so predictions are compared against each fit's own optimum instead of
    across fits.
    """
    X, y = load_iris(return_X_y=True)
    enc = ThresholdGuessBinarizer(n_estimators=10, max_depth=2, random_state=0)
    X_bin = enc.fit_transform(X, y)

    baseline = None
    for worker_limit in (1, 3, 0):
        clf = GOSDTClassifier(
            regularization=0.1, depth_budget=2, worker_limit=worker_limit
        )
        clf.fit(X_bin, y)
        result = clf.get_result()

        assert result["status"] == Status.CONVERGED
        # Same objective => same number of training misclassifications, so a
        # matching certified triple also pins the training accuracy.
        certified = (
            result["model_loss"],
            result["lower_bound"],
            result["upper_bound"],
            clf.score(X_bin, y),
        )
        if baseline is None:
            baseline = certified
        else:
            assert certified == pytest.approx(baseline)


def test_predict_before_fit_raises():
    clf = GOSDTClassifier()
    with pytest.raises(ValueError):
        clf.predict([[1, 0]])


def test_predict_model_number_validation():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    clf = GOSDTClassifier(regularization=0.05, allow_small_reg=True)
    clf.fit(X, y)

    with pytest.raises(ValueError):
        clf.predict(X, model_number=1)


def test_invalid_inputs():
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 0, 1])

    with pytest.raises(ValueError):
        GOSDTClassifier(regularization=-1.0)

    with pytest.raises(ValueError):
        GOSDTClassifier(depth_budget=-1)

    with pytest.raises(ValueError):
        GOSDTClassifier(upperbound_guess=1.5).fit(X, y)

    with pytest.raises(ValueError):
        GOSDTClassifier(time_limit=-1)


def test_time_limit_returns_partial():
    """A tiny time limit must not crash; a (possibly partial) model returns."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(400, 30)).astype(np.uint8)
    y = (X[:, 0] ^ X[:, 1] & X[:, 2] | X[:, 3]).astype(np.uint8)

    clf = GOSDTClassifier(regularization=0.001, allow_small_reg=True, time_limit=1)
    clf.fit(X, y)

    assert len(clf.trees_) >= 1
    assert clf.result_.status in (
        Status.CONVERGED,
        Status.TIMEOUT,
        Status.NON_CONVERGENCE,
    )
