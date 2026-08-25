"""Integration tests for minimal-flip feature tweaking (RashomonFlipSearch).

Covers SPOTSET (all-trees-in-the-Rashomon-set scope) and SPOT (single
optimal tree) adapters, cross-checking solver optimality against exhaustive
subset enumeration and verifying every returned tweak by re-prediction.
"""

import itertools

import numpy as np
import pytest

from shinrin import SPOTClassifier, SPOTSETClassifier

from shinrin.tweaking import RashomonFlipSearch, summarize_flip_results


@pytest.fixture(scope="module")
def xor_spotset():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, size=(60, 4)).astype(float)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    clf = SPOTSETClassifier(
        regularization=0.02, rashomon_bound_multiplier=0.35
    ).fit(X, y)
    return clf, X


def _brute_min_flips(trees, x, target_idx, max_size=4):
    for size in range(max_size + 1):
        for combo in itertools.combinations(range(x.size), size):
            xc = x.copy()
            for f in combo:
                xc[f] = 1.0 - xc[f]
            if all(
                t.predict(xc.reshape(1, -1))[0] == target_idx for t in trees
            ):
                return size, combo
    return None, None


def test_spotset_rashomon_scope_matches_brute_force(xor_spotset):
    clf, X = xor_spotset
    trees = [clf[i] for i in range(clf.n_trees_)]
    search = RashomonFlipSearch(clf)
    results = search.search(X, scope="rashomon")

    assert len(results) == X.shape[0]
    for i, res in enumerate(results):
        # default target = opposite of the reference tree's prediction
        ref_pred = int(trees[0].predict(X[i].reshape(1, -1))[0])
        assert int(res.target) != ref_pred
        expected, combo = _brute_min_flips(trees, X[i], int(res.target))
        if expected is None:
            assert not res.success and res.optimal
            continue
        assert res.success, f"row {i}: missed flip {combo}"
        assert res.l1_distance == pytest.approx(expected)
        assert res.verified
        assert res.optimal
        # every tree genuinely predicts the target after the tweak
        for t in trees:
            assert t.predict(res.x_new.reshape(1, -1))[0] == res.target


def test_reference_scope_never_exceeds_rashomon_distance(xor_spotset):
    clf, X = xor_spotset
    search = RashomonFlipSearch(clf)
    ref = search.search(X, scope="reference")
    rash = search.search(X, scope="rashomon")
    summary_ref = summarize_flip_results(ref)
    assert summary_ref["success_rate"] > 0.0
    for a, b in zip(ref, rash):
        if a.success and b.success:
            assert a.l1_distance <= b.l1_distance + 1e-12


def test_spot_single_tree_flip_verified():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(80, 4)).astype(float)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    spot = SPOTClassifier(regularization=0.02, allow_small_reg=True).fit(X, y)

    search = RashomonFlipSearch(spot)
    results = search.search(X[:30], scope="reference")
    for i, res in enumerate(results):
        assert res.success and res.verified and res.optimal
        assert res.n_models_total == 1
        # flipping exactly the reported columns must reproduce the target
        xc = X[i].copy()
        for f in res.changed_features:
            xc[f] = 1.0 - xc[f]
        assert spot.predict(xc.reshape(1, -1))[0] == res.target


def test_explicit_target_and_unknown_target_errors(xor_spotset):
    clf, X = xor_spotset
    search = RashomonFlipSearch(clf)
    out = search.search(X[:4], scope="reference", target=1)
    assert all(r.target == 1 for r in out)
    with pytest.raises(ValueError, match="not among classes"):
        search.search(X[:2], target="nope")
    with pytest.raises(ValueError, match="Unknown scope"):
        search.search(X[:2], scope="bogus")


def test_multiclass_requires_explicit_target():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    X_bin = (X > X.mean(axis=0)).astype(float)
    clf = SPOTSETClassifier(regularization=0.05, depth_budget=3).fit(X_bin, y)
    search = RashomonFlipSearch(clf)
    with pytest.raises(ValueError, match="explicit"):
        search.search(X_bin[:2])
    out = search.search(X_bin[:3], scope="reference", target=0)
    assert all(r.target == 0 for r in out)


def test_feature_count_mismatch_raises(xor_spotset):
    clf, X = xor_spotset
    search = RashomonFlipSearch(clf)
    with pytest.raises(ValueError, match="features"):
        search.search(X[:, :2])
