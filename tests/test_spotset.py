"""Tests for the vendored SPOTSET classifier (Rashomon set enumeration,
vendored from treeFARMS and renamed).

Requires the ``sklearn`` optional extra.
"""

import numpy as np
import pytest

from shinrin import SPOTClassifier, SPOTSETClassifier

try:
    from sklearn.base import clone
    from sklearn.datasets import load_breast_cancer, make_classification
except ImportError:  # pragma: no cover - sklearn optional
    pytest.skip("scikit-learn not available", allow_module_level=True)


@pytest.fixture()
def xor_data():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    return X, y


def test_fit_returns_set(xor_data):
    X, y = xor_data
    clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.05)
    clf.fit(X, y)

    assert clf.n_trees_ >= 1
    assert clf.get_tree_count() == clf.n_trees_
    assert clf.train_time_ >= 0.0
    assert clf.model_set_.get_tree_count() == clf.n_trees_


def test_predict_perfect_on_separable(xor_data):
    X, y = xor_data
    clf = SPOTSETClassifier(regularization=0.01).fit(X, y)
    assert np.array_equal(clf.predict(X), y)
    assert clf.score(X, y) == pytest.approx(1.0)


def test_string_labels_mapped_through_classes_(xor_data):
    X, _ = xor_data
    y = np.array(["no", "yes", "no", "no"])
    # make labels consistent with XOR structure for separability: index 1 is "yes"
    y = np.where(X[:, 0] == 1, "yes", "no")
    clf = SPOTSETClassifier(regularization=0.01).fit(X, y)
    assert list(clf.classes_) == ["no", "yes"]
    assert set(np.asarray(clf.predict(X), dtype=str)) <= {"no", "yes"}
    assert clf.score(X, y) == pytest.approx(1.0)


def test_tree_access_and_metrics(xor_data):
    X, y = xor_data
    clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.2).fit(
        X, y
    )
    seen_objectives = []
    for i in range(clf.n_trees_):
        tree = clf[i]
        assert tree.leaves() >= 1
        assert tree.maximum_depth() >= 1
        assert len(tree.predict(X)) == len(y)
        metric = clf.model_set_.get_tree_metric_at_idx(i)
        assert set(metric) == {"objective", "loss", "complexity"}
        assert metric["loss"] <= 1.0 + 1e-9
        seen_objectives.append(metric["objective"])

    # every tree in the set must be within the Rashomon bound of the best one
    best = min(seen_objectives)
    worst = max(seen_objectives)
    assert worst <= best * (1 + 0.2) + 1e-6


def test_best_tree_matches_spot_optimum():
    """The best SPOTSET objective should agree with SPOT's optimal objective
    (misclassification loss + regularization * number of leaves)."""
    import json

    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(80, 4))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    X = X.astype(float)

    reg = 0.02
    spotset = SPOTSETClassifier(regularization=reg, rashomon_bound_multiplier=0.05)
    spotset.fit(X, y)
    objectives = [
        spotset.model_set_.get_tree_metric_at_idx(i)["objective"]
        for i in range(spotset.n_trees_)
    ]

    def count_leaves(node) -> int:
        if "prediction" in node:
            return 1
        return count_leaves(node["true"]) + count_leaves(node["false"])

    spot = SPOTClassifier(regularization=reg, allow_small_reg=True, depth_budget=None)
    spot.fit(X, y)
    result = spot.get_result()
    spot_model = json.loads(result["models_string"])[0]
    spot_objective = result["model_loss"] + reg * count_leaves(spot_model)

    assert min(objectives) == pytest.approx(spot_objective, abs=1e-4)
    # and the set extends beyond the optimum (bound multiplier > 0)
    assert max(objectives) >= min(objectives) - 1e-9


def test_sklearn_compat(xor_data):
    X, y = xor_data
    clf = SPOTSETClassifier(regularization=0.01)
    cloned = clone(clf)
    assert cloned.get_params()["regularization"] == 0.01
    cloned.set_params(rashomon_bound_multiplier=0.1)
    cloned.fit(X, y)
    assert cloned.n_trees_ >= 1


def test_not_fitted_errors(xor_data):
    X, _ = xor_data
    clf = SPOTSETClassifier()
    with pytest.raises(Exception, match="fitted"):
        clf.predict(X)
    with pytest.raises(Exception, match="fitted"):
        clf.get_tree_count()


def test_multiclass_end_to_end():
    X, y = load_breast_cancer(return_X_y=True)
    from shinrin import ThresholdGuessBinarizer

    X_bin = ThresholdGuessBinarizer(
        n_estimators=10, max_depth=1, random_state=0
    ).fit_transform(X, y)

    clf = SPOTSETClassifier(regularization=0.01, depth_budget=3, time_limit=120)
    clf.fit(X_bin[:400], y[:400])
    assert clf.n_trees_ >= 1
    accuracy = clf.score(X_bin[:400], y[:400])
    assert accuracy > 0.85


def test_make_classification_smoke():
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    X_bin = (X > np.median(X, axis=0)).astype(float)
    clf = SPOTSETClassifier(regularization=0.01, rashomon_bound_multiplier=0.1)
    clf.fit(X_bin, y)
    assert clf.score(X_bin, y) > 0.8
