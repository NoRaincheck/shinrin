"""Parity tests between the Rust and Mojo native backends.

Both backends implement the same xorshift RNG and the same arithmetic, so
fitting with an identical ``random_state`` must produce identical trees.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from shinrin._backend import get_backend_module

pytest.importorskip("mojo", reason="Mojo backend required for parity tests")

# Arrays that must match bit-for-bit between backends.
TREE_ATTRS_EXACT = [
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "n_node_samples",
    "weighted_n_node_samples",
    "tau",
]

# Impurity/variance/value flow through multiply-add chains (running means,
# variance updates) that compilers may fuse differently (FMA), so allow
# last-ulp drift for these. Observed drift is ~1e-15 relative.
TREE_ATTRS_APPROX = ["impurity", "variance"]


def make_data(n_samples=120, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.randn(n_samples) * 0.1).astype(
        np.float64
    )
    return X, y


def fit_regression_tree(backend_module, X, y, seed=3, max_depth=6):
    tree = backend_module.Tree(X.shape[1], np.array([1], dtype=np.intp), 1)
    criterion = backend_module.MSE(1, X.shape[0])
    splitter = backend_module.MondrianSplitter(criterion, np.random.RandomState(seed))
    builder = backend_module.DepthFirstTreeBuilder(splitter, 2, max_depth)
    builder.build(tree, X, y.reshape(-1, 1))
    return tree


def fit_classification_tree(backend_module, X, y_bin, seed=3, max_depth=6):
    n_classes = len(np.unique(y_bin))
    tree = backend_module.Tree(X.shape[1], np.array([n_classes], dtype=np.intp), 1)
    criterion = backend_module.ClassificationCriterion(
        1, np.array([n_classes], dtype=np.intp)
    )
    splitter = backend_module.MondrianSplitter(criterion, np.random.RandomState(seed))
    builder = backend_module.DepthFirstTreeBuilder(splitter, 2, max_depth)
    builder.build(tree, X, y_bin.reshape(-1, 1).astype(np.float64))
    return tree


@pytest.fixture(scope="module")
def backends():
    rust = get_backend_module()  # default env is rust in CI/test runs

    import os

    old = os.environ.get("SHINRIN_BACKEND")
    os.environ["SHINRIN_BACKEND"] = "mojo"
    try:
        import shinrin._backend as backend_mod

        backend_mod._CACHE.clear()
        mojo = backend_mod.get_backend_module()
    finally:
        if old is None:
            os.environ.pop("SHINRIN_BACKEND", None)
        else:
            os.environ["SHINRIN_BACKEND"] = old
        backend_mod._CACHE.clear()
    return rust, mojo


def assert_trees_identical(rust_tree, mojo_tree):
    assert rust_tree.node_count == mojo_tree.node_count
    assert rust_tree.max_depth == mojo_tree.max_depth
    assert rust_tree.root == mojo_tree.root
    for attr in TREE_ATTRS_EXACT:
        a = getattr(rust_tree, attr)
        b = getattr(mojo_tree, attr)
        np.testing.assert_array_equal(a, b, err_msg=f"mismatch in {attr}")
    for attr in TREE_ATTRS_APPROX:
        a = getattr(rust_tree, attr)
        b = getattr(mojo_tree, attr)
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-12, err_msg=f"mismatch in {attr}"
        )
    np.testing.assert_array_equal(rust_tree.lower_bounds, mojo_tree.lower_bounds)
    np.testing.assert_array_equal(rust_tree.upper_bounds, mojo_tree.upper_bounds)
    np.testing.assert_allclose(rust_tree.value, mojo_tree.value, rtol=1e-9, atol=1e-12)


def test_regression_tree_structure_parity(backends):
    rust, mojo = backends
    X, y = make_data()
    rust_tree = fit_regression_tree(rust, X, y)
    mojo_tree = fit_regression_tree(mojo, X, y)
    assert_trees_identical(rust_tree, mojo_tree)


def test_regression_predictions_parity(backends):
    rust, mojo = backends
    X, y = make_data(200)
    rust_tree = fit_regression_tree(rust, X, y)
    mojo_tree = fit_regression_tree(mojo, X, y)

    mean_r, std_r = rust_tree.predict(X, return_std=True)
    mean_m, std_m = mojo_tree.predict(X, return_std=True)
    np.testing.assert_allclose(mean_r, mean_m, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(std_r, std_m, rtol=1e-12, atol=0.0)

    np.testing.assert_array_equal(rust_tree.apply(X), mojo_tree.apply(X))


@pytest.mark.parametrize("path_smoothing", [True, False])
def test_predict_mode_parity(backends, path_smoothing):
    """Both prediction modes must agree bit-for-bit across backends."""
    rust, mojo = backends
    X, y = make_data(200)

    # Scaled points land outside the training bounding boxes so the two
    # modes actually diverge.
    X_out = X * 3.0

    rust_reg = fit_regression_tree(rust, X, y)
    mojo_reg = fit_regression_tree(mojo, X, y)
    assert_trees_identical(rust_reg, mojo_reg)

    mean_r, std_r = rust_reg.predict(
        X_out, return_std=True, path_smoothing=path_smoothing
    )
    mean_m, std_m = mojo_reg.predict(
        X_out, return_std=True, path_smoothing=path_smoothing
    )
    np.testing.assert_allclose(mean_r, mean_m, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(std_r, std_m, rtol=1e-12, atol=0.0)

    # Constant mode must reproduce the raw leaf values exactly.
    if not path_smoothing:
        leaf_vals = np.array([rust_reg.value[nid][0] for nid in rust_reg.apply(X_out)])
        np.testing.assert_allclose(mean_r, leaf_vals.ravel(), rtol=1e-6)

    y_bin = (y > np.median(y)).astype(np.float64)
    rust_clf = fit_classification_tree(rust, X, y_bin)
    mojo_clf = fit_classification_tree(mojo, X, y_bin)
    assert_trees_identical(rust_clf, mojo_clf)

    proba_r = rust_clf.predict(
        X_out, is_regression=False, path_smoothing=path_smoothing
    )[0]
    proba_m = mojo_clf.predict(
        X_out, is_regression=False, path_smoothing=path_smoothing
    )[0]
    np.testing.assert_allclose(proba_r, proba_m, rtol=1e-12, atol=0.0)


def test_classification_parity(backends):
    rust, mojo = backends
    X, y_cont = make_data(150)
    y_bin = (y_cont > np.median(y_cont)).astype(np.float64)

    rust_tree = fit_classification_tree(rust, X, y_bin)
    mojo_tree = fit_classification_tree(mojo, X, y_bin)
    assert_trees_identical(rust_tree, mojo_tree)

    proba_r = rust_tree.predict(X, is_regression=False)[0]
    proba_m = mojo_tree.predict(X, is_regression=False)[0]
    np.testing.assert_allclose(proba_r, proba_m, rtol=1e-12, atol=0.0)


def test_partial_fit_parity(backends):
    rust, mojo = backends
    X, y = make_data(100)

    trees = []
    for mod in (rust, mojo):
        tree = mod.Tree(X.shape[1], np.array([1], dtype=np.intp), 1)
        builder = mod.PartialFitTreeBuilder(2, 10, np.random.RandomState(11))
        builder.build(tree, X[:40].copy(), y[:40].reshape(-1, 1))
        builder.build(tree, X[40:70].copy(), y[40:70].reshape(-1, 1))
        builder.build(tree, X[70:].copy(), y[70:].reshape(-1, 1))
        trees.append(tree)

    assert_trees_identical(trees[0], trees[1])
    np.testing.assert_allclose(
        trees[0].predict(X)[0], trees[1].predict(X)[0], rtol=1e-12, atol=0.0
    )


def test_decision_path_parity(backends):
    rust, mojo = backends
    X, y = make_data(80)
    rust_tree = fit_regression_tree(rust, X, y)
    mojo_tree = fit_regression_tree(mojo, X, y)

    path_r = rust_tree.decision_path(X).toarray()
    path_m = mojo_tree.decision_path(X).toarray()
    np.testing.assert_array_equal(path_r, path_m)

    wr_path_r = rust_tree.weighted_decision_path(X).toarray()
    wr_path_m = mojo_tree.weighted_decision_path(X).toarray()
    np.testing.assert_allclose(wr_path_r, wr_path_m, rtol=1e-9, atol=1e-12)

    np.testing.assert_array_equal(
        rust_tree.isolation_path_length(X), mojo_tree.isolation_path_length(X)
    )


def test_shap_values_parity(backends):
    rust, mojo = backends
    X, y = make_data(60)
    rust_tree = fit_regression_tree(rust, X, y)
    mojo_tree = fit_regression_tree(mojo, X, y)
    np.testing.assert_allclose(
        rust_tree.shap_values(X), mojo_tree.shap_values(X), rtol=1e-12, atol=0.0
    )


def test_pickle_roundtrip_mojo_backend():
    mojo = _load_mojo_fresh()
    X, y = make_data(90)
    tree = fit_regression_tree(mojo, X, y)
    expected = tree.predict(X)[0]

    restored = pickle.loads(pickle.dumps(tree))
    assert restored.node_count == tree.node_count
    for attr in TREE_ATTRS_EXACT:
        np.testing.assert_array_equal(getattr(tree, attr), getattr(restored, attr))
    np.testing.assert_allclose(restored.predict(X)[0], expected, rtol=1e-12)


def _load_mojo_fresh():
    import os

    import shinrin._backend as backend_mod

    old = os.environ.get("SHINRIN_BACKEND")
    os.environ["SHINRIN_BACKEND"] = "mojo"
    try:
        backend_mod._CACHE.clear()
        return backend_mod.get_backend_module()
    finally:
        if old is None:
            os.environ.pop("SHINRIN_BACKEND", None)
        else:
            os.environ["SHINRIN_BACKEND"] = old
        backend_mod._CACHE.clear()


def _activate_backend(backend):
    """Point the public API at a backend by reloading the shim modules."""
    import importlib
    import os

    import shinrin
    import shinrin._backend as backend_mod
    import shinrin._skgarden.mondrian.tree._criterion as crit_shim
    import shinrin._skgarden.mondrian.tree._splitter as splitter_shim
    import shinrin._skgarden.mondrian.tree._tree as tree_shim

    os.environ["SHINRIN_BACKEND"] = backend
    backend_mod._CACHE.clear()
    importlib.reload(crit_shim)
    importlib.reload(splitter_shim)
    importlib.reload(tree_shim)
    shinrin._cache.clear()


def test_forest_end_to_end_parity():
    """Full MondrianForestRegressor fit/predict parity through the public API."""
    import os

    import shinrin

    X, y = make_data(100, seed=42)
    results = []
    try:
        for backend in ("rust", "mojo"):
            _activate_backend(backend)
            model = shinrin.MondrianForestRegressor(n_estimators=3, random_state=7)
            model.fit(X, y)
            results.append(model.predict(X))
    finally:
        default = os.environ.get("SHINRIN_BACKEND", "rust")
        _activate_backend(default)
        if default == "rust":
            os.environ.pop("SHINRIN_BACKEND", None)

    np.testing.assert_allclose(results[0], results[1], rtol=1e-12, atol=0.0)
