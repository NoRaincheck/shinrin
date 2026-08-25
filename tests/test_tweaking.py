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
    clf = SPOTSETClassifier(regularization=0.02, rashomon_bound_multiplier=0.35).fit(
        X, y
    )
    return clf, X


def _brute_min_flips(trees, x, target_idx, max_size=4):
    for size in range(max_size + 1):
        for combo in itertools.combinations(range(x.size), size):
            xc = x.copy()
            for f in combo:
                xc[f] = 1.0 - xc[f]
            if all(t.predict(xc.reshape(1, -1))[0] == target_idx for t in trees):
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
        assert res.x_new is not None
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


# ---------------------------------------------------------------------------
# scikit-learn forest / tree adapters (continuous features, L1 tweaks)
# ---------------------------------------------------------------------------


def _tree_min_flip_independent(tree, x, target_idx):
    """Reference minimal L1 flip for one sklearn tree, written independently."""
    INF = float("inf")
    best = INF
    stack = [(0, {})]
    while stack:
        node, path = stack.pop()
        if tree.children_left[node] == -1:
            if int(np.argmax(tree.value[node])) != target_idx:
                continue
            best = min(
                best,
                sum(max(0.0, lo - x[f], x[f] - hi) for f, (lo, hi) in path.items()),
            )
            continue
        f, t = int(tree.feature[node]), float(tree.threshold[node])
        plo, phi = path.get(f, (-np.inf, np.inf))
        lo_path = dict(path)
        lo_path[f] = (plo, min(phi, t))
        hi_path = dict(path)
        hi_path[f] = (max(plo, np.nextafter(t, np.inf)), phi)
        stack.append((int(tree.children_left[node]), lo_path))
        stack.append((int(tree.children_right[node]), hi_path))
    return best


@pytest.fixture(scope="module")
def small_forest_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        class_sep=1.0,
        random_state=0,
    )
    return X, y


def test_sklearn_single_tree_minimality(small_forest_data):
    from sklearn.tree import DecisionTreeClassifier

    X, y = small_forest_data
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    search = RashomonFlipSearch(tree)
    results = search.search(X[:25], scope="reference", target=int(y[0]) ^ 1)
    for i, res in enumerate(results):
        expected = _tree_min_flip_independent(tree.tree_, X[i], int(res.target))
        assert res.success and res.verified
        # adapter boundaries are float32-safe (sklearn casts inputs to
        # float32), so distances can differ from raw-float64 arithmetic by a
        # representable-ULP-sized sliver
        assert res.l1_distance == pytest.approx(expected, abs=1e-5)


def test_sklearn_forest_rashomon_scope(small_forest_data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = small_forest_data
    forest = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0).fit(
        X, y
    )
    search = RashomonFlipSearch(forest)

    reference = search.search(X[:20], scope="reference")
    assert all(r.success and r.verified and r.n_models_total == 8 for r in reference)

    robust = search.search(X[:20], scope="rashomon", max_nodes=200_000)
    solved = [r for r in robust if r.success]
    assert solved, "expected at least one solvable all-trees tweak"
    for ref_res, rob_res in zip(reference, robust):
        if ref_res.success and rob_res.success:
            assert rob_res.l1_distance >= ref_res.l1_distance - 1e-12
        if rob_res.success:
            assert rob_res.verified
            assert rob_res.x_new is not None
            preds = forest.predict(rob_res.x_new.reshape(1, -1))[0]
            assert preds == rob_res.target


# ---------------------------------------------------------------------------
# scope="ensemble": every sklearn tree family via its own aggregation rule
# ---------------------------------------------------------------------------


def test_ensemble_soft_vote_forest_matches_predict(small_forest_data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = small_forest_data
    forest = RandomForestClassifier(n_estimators=9, max_depth=4, random_state=0).fit(
        X, y
    )
    results = RashomonFlipSearch(forest).search(X[:20], scope="ensemble", target=1)
    for res in results:
        assert res.verified or not res.success
        if res.success:
            assert res.x_new is not None
            assert forest.predict(res.x_new.reshape(1, -1))[0] == 1


def test_ensemble_single_tree_equals_reference_minimality(small_forest_data):
    from sklearn.tree import DecisionTreeClassifier

    X, y = small_forest_data
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    search = RashomonFlipSearch(tree)
    ensemble = search.search(X[:15], scope="ensemble", target=int(y[0]) ^ 1)
    reference = search.search(X[:15], scope="reference", target=int(y[0]) ^ 1)
    # for a single soft-voting tree the aggregate rule reduces to the leaf
    # majority, so both scopes share the same optimum
    for ens_res, ref_res in zip(ensemble, reference):
        assert ens_res.success and ref_res.success
        assert ens_res.l1_distance == pytest.approx(ref_res.l1_distance)


def test_ensemble_adaboost_weighted_vote(small_forest_data):
    from sklearn.ensemble import AdaBoostClassifier

    X, y = small_forest_data
    ada = AdaBoostClassifier(n_estimators=8, random_state=0).fit(X, y)
    results = RashomonFlipSearch(ada).search(X[:20], scope="ensemble", target=1)
    solved = [r for r in results if r.success]
    assert solved
    for res in solved:
        assert res.verified and res.x_new is not None
        assert ada.predict(res.x_new.reshape(1, -1))[0] == 1


def test_gbm_leaf_values_reconstruct_decision_function(small_forest_data):
    from sklearn.ensemble import GradientBoostingClassifier

    from shinrin._tweaking._sklearn_adapter import GradientBoostingView

    X, y = small_forest_data
    gbm = GradientBoostingClassifier(n_estimators=8, max_depth=2, random_state=0).fit(
        X[:200], y[:200]
    )
    view = GradientBoostingView(gbm)
    recon = view.base_score + view.score_matrix(X).sum(axis=0)
    assert np.allclose(recon, gbm.decision_function(X).ravel())


def test_hgbt_leaves_reconstruct_decision_function(small_forest_data):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from shinrin._tweaking._sklearn_adapter import HistGradientBoostingView

    X, y = small_forest_data
    hgb = HistGradientBoostingClassifier(max_iter=8, max_depth=2, random_state=0).fit(
        X, y
    )
    view = HistGradientBoostingView(hgb)
    recon = view.base_score + view.score_matrix(X).sum(axis=0)
    assert np.allclose(recon, hgb.decision_function(X).ravel())


def test_boosted_ensembles_flip_verified(small_forest_data):
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )

    X, y = small_forest_data
    for make in (
        lambda: GradientBoostingClassifier(
            n_estimators=10, max_depth=2, random_state=0
        ),
        lambda: HistGradientBoostingClassifier(
            max_iter=10, max_depth=2, random_state=0
        ),
    ):
        model = make().fit(X, y)
        results = RashomonFlipSearch(model).search(X[:15], scope="ensemble", target=1)
        solved = [r for r in results if r.success]
        assert solved
        for res in solved:
            assert res.verified and res.x_new is not None
            assert model.predict(res.x_new.reshape(1, -1))[0] == 1


def test_boosted_multiclass_rejected():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )

    X, y = load_iris(return_X_y=True)
    with pytest.raises(NotImplementedError):
        RashomonFlipSearch(GradientBoostingClassifier(n_estimators=5).fit(X, y))
    with pytest.raises(NotImplementedError):
        RashomonFlipSearch(HistGradientBoostingClassifier(max_iter=5).fit(X, y))


def test_rashomon_scope_on_boosting_rejected(small_forest_data):
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = small_forest_data
    gb = GradientBoostingClassifier(n_estimators=5, max_depth=1, random_state=0).fit(
        X, y
    )
    with pytest.raises(ValueError, match="boosted"):
        RashomonFlipSearch(gb).search(X[:2], scope="rashomon")
