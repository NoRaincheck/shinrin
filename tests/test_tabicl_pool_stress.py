"""Determinism stress tests for the mojo TabICL backend's worker pool.

The pool reuses parked pthread workers across parallel regions; these tests
hammer varied shapes and repeated calls, asserting bit-identical outputs
across runs (any scheduling-dependent divergence - races, unsynchronized
scratch, stale snapshots - shows up as a mismatch).

Gated by ``SHINRIN_TABICL_PARITY_MOJO=1`` like the other mojo tests.
"""

import os

import numpy as np
import pytest

mojo_enabled = os.environ.get("SHINRIN_TABICL_PARITY_MOJO") == "1"

pytestmark = pytest.mark.skipif(
    not mojo_enabled, reason="requires SHINRIN_TABICL_PARITY_MOJO=1"
)


def _make(seed: int, n: int, f: int):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, f).astype(np.float32)
    y = (X @ rng.randn(f) > 0).astype(np.int64)
    return X, y


def _predict_twice(
    n_train: int,
    n_test: int,
    n_features: int,
    seed: int,
    estimators: int,
    batch_size: int,
    repeats: int = 3,
):
    from shinrin.tabicl import TabICLClassifier

    X, y = _make(seed, n_train + n_test, n_features)
    clf = TabICLClassifier(
        backend="mojo",
        n_estimators=estimators,
        batch_size=batch_size,
        random_state=42,
        kv_cache=False,
    ).fit(X[:n_train], y[:n_train])
    X_test = X[n_train:]

    first = None
    for _ in range(repeats):
        out = clf.predict_proba(X_test)
        if first is None:
            first = out
        else:
            np.testing.assert_array_equal(first, out)
    return first


def test_mojo_pool_determinism_small_shapes():
    # Many small parallel regions: worst case for stale-snapshot races.
    for seed, n_train, n_test, n_features in [
        (0, 120, 60, 7),
        (1, 300, 50, 10),
        (2, 97, 43, 5),  # odd sizes -> uneven row partitions
    ]:
        _predict_twice(n_train, n_test, n_features, seed, estimators=4, batch_size=200)


def test_mojo_pool_determinism_batch_cliff():
    # Tiny batch sizes force many chunked predict passes per call.
    _predict_twice(400, 96, 12, seed=3, estimators=4, batch_size=8)
    _predict_twice(400, 96, 12, seed=3, estimators=4, batch_size=16)


def test_mojo_pool_determinism_repeated_models():
    # Fresh model per iteration: exercises repeated pool create/shutdown.
    for i in range(4):
        _predict_twice(200, 40, 8, seed=10 + i, estimators=2, batch_size=100)
