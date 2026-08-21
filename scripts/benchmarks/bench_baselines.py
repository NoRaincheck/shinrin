#!/usr/bin/env python3
"""Benchmark Shinrin against LightGBM and scikit-learn SGD baselines.

Compares training time, prediction time, and partial fit performance
for regression and classification tasks. For backend (Rust vs Mojo)
comparisons, see bench_backends.py.

Usage:
    python scripts/benchmarks/bench_baselines.py
"""

from __future__ import annotations

import time

import numpy as np

from shinrin import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    MondrianTreeClassifier,
    MondrianTreeRegressor,
)

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed, skipping LightGBM benchmarks")

try:
    from sklearn.linear_model import SGDClassifier, SGDRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed, skipping SGD benchmarks")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def _regression_bench() -> None:
    """Run regression benchmarks."""
    print("=" * 70)
    print("REGRESSION BENCHMARKS (5000 samples, 20 features)")
    print("=" * 70)

    X_train = np.random.randn(5000, 20).astype(np.float32)
    y_train = (
        np.sum(X_train[:, :5], axis=1) + np.random.randn(5000).astype(np.float32) * 0.1
    )
    X_test = np.random.randn(1000, 20).astype(np.float32)

    # --- Train ---
    print("\nTraining Time:")

    t0 = time.perf_counter()
    m = MondrianTreeRegressor(max_depth=8, random_state=0)
    m.fit(X_train, y_train)
    shinrin_tree_train = time.perf_counter() - t0
    print(f"  Shinrin Tree (depth=8):      {shinrin_tree_train:.4f}s")

    t0 = time.perf_counter()
    m = MondrianForestRegressor(n_estimators=10, max_depth=8, random_state=0)
    m.fit(X_train, y_train)
    shinrin_forest_train = time.perf_counter() - t0
    print(f"  Shinrin Forest (n=10):       {shinrin_forest_train:.4f}s")

    if HAS_LGB:
        t0 = time.perf_counter()
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "regression", "num_leaves": 31, "verbose": -1}
        lgb.train(params, dtrain, num_boost_round=8)
        lgb_tree_train = time.perf_counter() - t0
        print(f"  LightGBM Tree (8 rounds):    {lgb_tree_train:.4f}s")

        t0 = time.perf_counter()
        dtrain = lgb.Dataset(X_train, label=y_train)
        lgb.train(params, dtrain, num_boost_round=10)
        lgb_forest_train = time.perf_counter() - t0
        print(f"  LightGBM Forest (10 rounds): {lgb_forest_train:.4f}s")

    if HAS_SKLEARN:
        t0 = time.perf_counter()
        sgd = SGDRegressor(max_iter=100, tol=1e-3, random_state=0)
        sgd.fit(X_train, y_train)
        sgd_train = time.perf_counter() - t0
        print(f"  SGDRegressor (100 iters):    {sgd_train:.4f}s")

    # --- Predict ---
    print("\nPrediction Time (1000 samples, 100 iterations):")

    m_tree = MondrianTreeRegressor(max_depth=8, random_state=0)
    m_tree.fit(X_train, y_train)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = m_tree.predict(X_test)
    shinrin_tree_pred = (time.perf_counter() - t0) / 100
    print(f"  Shinrin Tree:                {shinrin_tree_pred * 1000:.4f}ms/call")

    m_forest = MondrianForestRegressor(n_estimators=10, max_depth=8, random_state=0)
    m_forest.fit(X_train, y_train)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = m_forest.predict(X_test)
    shinrin_forest_pred = (time.perf_counter() - t0) / 100
    print(f"  Shinrin Forest:              {shinrin_forest_pred * 1000:.4f}ms/call")

    if HAS_LGB:
        lgb_model = lgb.train(params, dtrain, num_boost_round=8)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = lgb_model.predict(X_test)
        lgb_tree_pred = (time.perf_counter() - t0) / 100
        print(f"  LightGBM Tree:               {lgb_tree_pred * 1000:.4f}ms/call")

    if HAS_SKLEARN:
        t0 = time.perf_counter()
        for _ in range(100):
            _ = sgd.predict(X_test)
        sgd_pred = (time.perf_counter() - t0) / 100
        print(f"  SGDRegressor:                {sgd_pred * 1000:.4f}ms/call")

    # --- Partial Fit ---
    print("\nPartial Fit (100 epochs, 5000 samples):")
    if HAS_SKLEARN:
        sgd2 = SGDRegressor(max_iter=1, tol=None, random_state=0)
        t0 = time.perf_counter()
        for _ in range(100):
            sgd2.partial_fit(X_train, y_train)
        sgd_partial = time.perf_counter() - t0
        print(f"  SGDRegressor:                {sgd_partial:.4f}s")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classification_bench() -> None:
    """Run classification benchmarks."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION BENCHMARKS (5000 samples, 20 features)")
    print("=" * 70)

    X_train = np.random.randn(5000, 20).astype(np.float32)
    y_train = (np.sum(X_train[:, :5], axis=1) > 0).astype(int)
    X_test = np.random.randn(1000, 20).astype(np.float32)

    # --- Train ---
    print("\nTraining Time:")

    t0 = time.perf_counter()
    m = MondrianTreeClassifier(max_depth=8, random_state=0)
    m.fit(X_train, y_train)
    shinrin_tree_train = time.perf_counter() - t0
    print(f"  Shinrin Tree (depth=8):      {shinrin_tree_train:.4f}s")

    t0 = time.perf_counter()
    m = MondrianForestClassifier(n_estimators=10, max_depth=8, random_state=0)
    m.fit(X_train, y_train)
    shinrin_forest_train = time.perf_counter() - t0
    print(f"  Shinrin Forest (n=10):       {shinrin_forest_train:.4f}s")

    if HAS_LGB:
        t0 = time.perf_counter()
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "binary", "verbose": -1}
        lgb.train(params, dtrain, num_boost_round=8)
        lgb_tree_train = time.perf_counter() - t0
        print(f"  LightGBM Tree (8 rounds):    {lgb_tree_train:.4f}s")

        t0 = time.perf_counter()
        dtrain = lgb.Dataset(X_train, label=y_train)
        lgb.train(params, dtrain, num_boost_round=10)
        lgb_forest_train = time.perf_counter() - t0
        print(f"  LightGBM Forest (10 rounds): {lgb_forest_train:.4f}s")

    if HAS_SKLEARN:
        t0 = time.perf_counter()
        sgd = SGDClassifier(max_iter=100, tol=1e-3, random_state=0)
        sgd.fit(X_train, y_train)
        sgd_train = time.perf_counter() - t0
        print(f"  SGDClassifier (100 iters):   {sgd_train:.4f}s")

    # --- Predict ---
    print("\nPrediction Time (1000 samples, 100 iterations):")

    m_tree = MondrianTreeClassifier(max_depth=8, random_state=0)
    m_tree.fit(X_train, y_train)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = m_tree.predict(X_test)
    shinrin_tree_pred = (time.perf_counter() - t0) / 100
    print(f"  Shinrin Tree:                {shinrin_tree_pred * 1000:.4f}ms/call")

    m_forest = MondrianForestClassifier(n_estimators=10, max_depth=8, random_state=0)
    m_forest.fit(X_train, y_train)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = m_forest.predict(X_test)
    shinrin_forest_pred = (time.perf_counter() - t0) / 100
    print(f"  Shinrin Forest:              {shinrin_forest_pred * 1000:.4f}ms/call")

    if HAS_LGB:
        lgb_model = lgb.train(params, dtrain, num_boost_round=8)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = lgb_model.predict(X_test)
        lgb_tree_pred = (time.perf_counter() - t0) / 100
        print(f"  LightGBM Tree:               {lgb_tree_pred * 1000:.4f}ms/call")

    if HAS_SKLEARN:
        t0 = time.perf_counter()
        for _ in range(100):
            _ = sgd.predict(X_test)
        sgd_pred = (time.perf_counter() - t0) / 100
        print(f"  SGDClassifier:               {sgd_pred * 1000:.4f}ms/call")

    # --- Partial Fit ---
    print("\nPartial Fit (100 epochs, 5000 samples):")
    if HAS_SKLEARN:
        classes = np.array([0, 1])
        sgd2 = SGDClassifier(max_iter=1, tol=None, random_state=0)
        t0 = time.perf_counter()
        for _ in range(100):
            sgd2.partial_fit(X_train, y_train, classes=classes)
        sgd_partial = time.perf_counter() - t0
        print(f"  SGDClassifier:               {sgd_partial:.4f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all benchmarks."""
    _regression_bench()
    _classification_bench()
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
