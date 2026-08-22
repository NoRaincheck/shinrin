#!/usr/bin/env python3
"""Benchmark TabICL inference: NumPy vs torch backends.

Times ``TabICLClassifier`` / ``TabICLRegressor`` fit (preprocessing +
optional KV-cache build) and predict (ensemble forward passes) separately
on synthetic datasets. The upstream ``tabicl`` package can be compared
with ``--with-upstream`` when the benchmark extra is installed:

    uv sync --extra tabicl-bench

The Mojo backend is **not** benchmarked yet: the native kernels are an
experimental scaffold without numeric parity (see
``scripts/benchmarks/TABICL_BENCHMARK.md``).

Usage:
    python scripts/benchmarks/bench_tabicl.py [--quick] [--repeat N]
        [--with-upstream] [--kv-cache]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from shinrin.tabicl import TabICLClassifier, TabICLRegressor


def make_regression(n_samples: int, n_features: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features).astype(np.float32)
    y = X @ w + 0.1 * rng.randn(n_samples).astype(np.float32)
    return X, y


def make_classification(n_samples: int, n_features: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    logits = X @ rng.randn(n_features).astype(np.float32)
    centers = np.linspace(-2.0, 2.0, 5)
    y = np.digitize(logits + 0.5 * rng.randn(n_samples), bins=centers)
    return X, y.astype(np.int64)


def make_mixed_classification(n_samples: int, n_features: int, seed: int = 2):
    """Continuous + low-cardinality 'categorical' columns."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    n_cat = max(1, n_features // 4)
    for j in range(n_cat):
        levels = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n_samples)
        X[:, j] = levels
    logits = (
        X[:, :n_cat].sum(axis=1)
        + 0.5 * X[:, n_cat:].sum(axis=1)
        + 0.3 * rng.randn(n_samples)
    )
    y = (logits > np.median(logits)).astype(np.int64)
    return X, y


def _timed(fn, repeats: int, warmup: bool = True):
    if warmup:
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return times.mean(), times.std()


def bench_case(task: str, n_samples: int, n_features: int, args) -> None:
    backend = "numpy" if not args.torch else "torch"
    n_test = min(1000, max(200, n_samples // 10))
    if task == "regression":
        X, y = make_regression(n_samples + n_test, n_features)
    elif task == "classification":
        X, y = make_classification(n_samples + n_test, n_features)
    else:
        X, y = make_mixed_classification(n_samples + n_test, n_features)
    X_train, y_train, X_test = X[:-n_test], y[:-n_test], X[-n_test:]

    common: dict[str, Any] = {
        "backend": backend,
        "n_estimators": args.n_estimators,
        "random_state": 42,
        "kv_cache": args.kv_cache,
        "batch_size": args.batch_size,
    }
    model = (
        TabICLRegressor(**common) if task == "regression" else TabICLClassifier(**common)
    )

    fit_time = predict_time = fit_std = predict_std = float("nan")
    score = float("nan")
    try:
        fit_time, fit_std = _timed(lambda: model.fit(X_train, y_train), args.repeat)

        def predict():
            if task == "regression":
                return model.predict(X_test)
            return model.predict_proba(X_test)

        predict_time, predict_std = _timed(predict, args.repeat)
        score = model.score(X_test, y[-n_test:])
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        print(f"  {task:>18} FAILED: {exc}")
        return

    print(
        f"  {task:>18} {n_samples:>6}x{n_features:<4} "
        f"fit {fit_time:7.3f}s ±{fit_std:.3f}  "
        f"predict {predict_time:7.3f}s ±{predict_std:.3f}  "
        f"score {score:.3f}  [{backend}]"
    )


def bench_upstream(task: str, n_samples: int, n_features: int, args) -> None:
    try:
        from tabicl import TabICLClassifier as UpstreamClassifier
        from tabicl import TabICLRegressor as UpstreamRegressor
    except ImportError:
        print("  upstream tabicl not installed; skipping")
        return
    n_test = min(1000, max(200, n_samples // 10))
    if task == "regression":
        X, y = make_regression(n_samples + n_test, n_features)
        model = UpstreamRegressor()
    elif task == "classification":
        X, y = make_classification(n_samples + n_test, n_features)
        model = UpstreamClassifier()
    else:
        X, y = make_mixed_classification(n_samples + n_test, n_features)
        model = UpstreamClassifier()
    X_train, y_train, X_test = X[:-n_test], y[:-n_test], X[-n_test:]
    try:
        fit_time, fit_std = _timed(lambda: model.fit(X_train, y_train), args.repeat)

        def predict():
            return model.predict(X_test)

        predict_time, predict_std = _timed(predict, args.repeat)
        score = model.score(X_test, y[-n_test:])
    except Exception as exc:  # noqa: BLE001
        print(f"  {task:>18} upstream FAILED: {exc}")
        return
    print(
        f"  {task:>18} {n_samples:>6}x{n_features:<4} "
        f"fit {fit_time:7.3f}s ±{fit_std:.3f}  "
        f"predict {predict_time:7.3f}s ±{predict_std:.3f}  "
        f"score {score:.3f}  [upstream]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small grid")
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats")
    parser.add_argument(
        "--n-estimators", type=int, default=8, help="ensemble members"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument(
        "--torch", action="store_true", help="benchmark the torch backend"
    )
    parser.add_argument(
        "--with-upstream",
        action="store_true",
        help="also compare against pip tabicl (needs tabicl-bench extra)",
    )
    args = parser.parse_args()

    if args.quick:
        sizes = [(300, 10)]
    else:
        sizes = [(300, 10), (1000, 10), (1000, 100), (5000, 100)]
    tasks = ["classification", "regression", "mixed categorical"]

    print(f"TabICL benchmark (backend={'torch' if args.torch else 'numpy'}, "
          f"n_estimators={args.n_estimators}, kv_cache={args.kv_cache})")
    for n_samples, n_features in sizes:
        print(f"\n--- dataset {n_samples} x {n_features} ---")
        for task in tasks:
            bench_case(task, n_samples, n_features, args)
            if args.with_upstream:
                bench_upstream(task, n_samples, n_features, args)


if __name__ == "__main__":
    main()
