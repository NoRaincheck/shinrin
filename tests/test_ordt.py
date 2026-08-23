import numpy as np
import pytest
from sklearn.datasets import make_classification

from shinrin import OrdtClassifier


def _data(n_samples: int = 300, seed: int = 0):
    return make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=seed,
    )


def test_ordt_fit_predict():
    pytest.importorskip("pandas")
    X, y = _data()
    clf = OrdtClassifier(n_estimators=5, max_depth=2, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})
    assert clf.score(X, y) > 0.6

    # pool bookkeeping is consistent
    assert set(clf.stats_) == {"mined", "usable", "selected"}
    assert 0 < clf.stats_["selected"] <= clf.stats_["usable"] <= clf.stats_["mined"]
    assert len(clf.pool_labels_) == len(clf.pool_rules_)
    clauses, n_rules = clf.complexity()
    assert n_rules >= 1 and clauses >= n_rules
    # mine/select timings recorded by fit
    assert clf.mine_s_ > 0 and clf.select_s_ >= 0


def test_ordt_list_rules_references_pool():
    pytest.importorskip("pandas")
    X, y = _data()
    clf = OrdtClassifier(n_estimators=5, max_depth=2, random_state=0)
    clf.fit(X, y)
    listed = clf.list_rules()
    _, n_rules = clf.complexity()
    assert len(listed) == n_rules
    for label, prediction in listed:
        assert isinstance(prediction, bool)
        if label.startswith("NOT "):
            assert label[4:] in clf.pool_labels_
        else:
            assert label in clf.pool_labels_


def test_ordt_deterministic_given_random_state():
    pytest.importorskip("pandas")
    X, y = _data()
    pred_a = OrdtClassifier(n_estimators=5, max_depth=2, random_state=7).fit(X, y)
    pred_b = OrdtClassifier(n_estimators=5, max_depth=2, random_state=7).fit(X, y)
    assert np.array_equal(pred_a.predict(X), pred_b.predict(X))
    assert pred_a.list_rules() == pred_b.list_rules()


def test_ordt_requires_binary_targets():
    pytest.importorskip("pandas")
    X, _ = _data(n_samples=60)
    y = np.full(60, 2)
    with pytest.raises(ValueError, match="binary targets"):
        OrdtClassifier(random_state=0).fit(X, y)


def test_ordt_empty_pool_raises():
    pytest.importorskip("pandas")
    X, y = _data()
    # an impossible precision threshold guarantees zero surviving candidates
    clf = OrdtClassifier(precision_min=1.01, random_state=0)
    with pytest.raises(RuntimeError, match="no usable rules"):
        clf.fit(X, y)
