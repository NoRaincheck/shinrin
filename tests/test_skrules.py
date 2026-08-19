import numpy as np
import pytest
from sklearn.datasets import make_classification

from shinrin import SkopeRules


def test_skope_rules_fit_predict():
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    sr = SkopeRules(
        n_estimators=5,
        max_depth=2,
        max_samples=0.8,
        random_state=0,
    )
    sr.fit(X, y)
    pred = sr.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})

    scores = sr.decision_function(X)
    assert scores.shape == (X.shape[0],)
    votes = sr.rules_vote(X)
    assert votes.shape == (X.shape[0],)
    top = sr.score_top_rules(X)
    assert top.shape == (X.shape[0],)
    top_pred = sr.predict_top_rules(X, 3)
    assert top_pred.shape == (X.shape[0],)

    assert len(sr.rules_) > 0
    assert len(sr.estimators_) == 2 * 5  # classifier + regressor baggings


def test_skope_rules_feature_names():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        random_state=1,
    )
    names = ["age", "income", "debt", "score"]
    sr = SkopeRules(n_estimators=2, max_depth=2, feature_names=names, random_state=0)
    sr.fit(X, y)
    for rule, _ in sr.rules_:
        assert isinstance(rule, str)
    joined = " ".join(rule for rule, _ in sr.rules_)
    assert "undefined!" not in joined


def test_skope_rules_requires_two_classes():
    X = np.zeros((10, 3))
    y = np.ones(10)
    sr = SkopeRules(random_state=0)
    with pytest.raises(ValueError):
        sr.fit(X, y)
