#!/usr/bin/env python3
"""Benchmark the shinrin MLP estimators against scikit-learn.

Times ``MLPRegressor`` / ``MLPClassifier`` fits on synthetic datasets for

- ``sklearn``  : scikit-learn's MLPRegressor / MLPClassifier (reference)
- ``numpy``    : shinrin MLP with the pure NumPy backend
- ``mojo``     : shinrin MLP with the Mojo kernels (``just build-mlp-mojo``)
- ``+PLE``     : shinrin MLP with piecewise-linear embeddings enabled
                 (asinh + standardize + PLE recipe; Mojo when available)

Usage:
    python scripts/benchmarks/bench_mlp.py [--samples N] [--features N]
        [--max-iter N] [--backends sklearn,numpy,mojo,ple]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from shinrin.mlp import MLPClassifier, MLPRegressor


def make_regression(n_samples: int, n_features: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features).astype(np.float32)
    y = X @ w * 20.0 + 2.0 * rng.randn(n_samples).astype(np.float32)
    # Standardized target keeps every implementation in a healthy regime.
    return X, ((y - y.mean()) / y.std()).astype(np.float32)


def make_classification(n_samples: int, n_features: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    logits = X @ rng.randn(n_features).astype(np.float32)
    y = (logits + 0.5 * rng.randn(n_samples)).argsort().argsort() % 3
    return X, y.astype(np.int64)


def make_mixed_classification(n_samples: int, n_features: int, seed: int = 2):
    """Continuous + low-cardinality columns (tabular-ish)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    n_cat = max(1, n_features // 3)
    for j in range(n_cat):
        X[:, j] = np.round(X[:, j] * 2.0) / 2.0  # ~6 levels
    logits = (
        2.0 * X[:, :n_cat].sum(axis=1)
        + 0.5 * X[:, n_cat:].sum(axis=1)
        + 0.3 * rng.randn(n_samples)
    )
    y = (logits > np.median(logits)).astype(np.int64)
    return X, y


def _kwargs(ple: bool):
    if not ple:
        return {}
    return {
        "use_embeddings": True,
        "use_asinh": True,
        "use_scaler": True,
    }


def bench_case(
    label: str,
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    ple: bool,
):
    if kind == "regression":
        model = MLPRegressor(max_iter=max_iter, random_state=0, **_kwargs(ple))
        ref = None
    else:
        model = MLPClassifier(max_iter=max_iter, random_state=0, **_kwargs(ple))
        ref = None
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    score = model.score(X, y)
    print(
        f"  {label:22s} fit {elapsed:7.2f}s  "
        f"score {score:.4f}  ({X.shape[0]}x{X.shape[1]}, {max_iter} epochs)"
    )
    return elapsed, score


def bench_sklearn(kind: str, X: np.ndarray, y: np.ndarray, max_iter: int):
    from sklearn.neural_network import MLPClassifier as SkClf
    from sklearn.neural_network import MLPRegressor as SkReg

    model = SkReg(max_iter=max_iter, random_state=0) if kind == "regression" else SkClf(
        max_iter=max_iter, random_state=0
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    print(
        f"  {'sklearn':22s} fit {elapsed:7.2f}s  "
        f"score {model.score(X, y):.4f}  ({X.shape[0]}x{X.shape[1]}, {max_iter} epochs)"
    )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--backends",
        default="sklearn,numpy,mojo,ple",
        help="comma list from: sklearn,numpy,mojo,ple",
    )
    args = parser.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    cases = [
        ("regression", make_regression(args.samples, args.features)),
        ("multiclass", make_classification(args.samples, args.features)),
        ("mixed categorical", make_mixed_classification(args.samples, args.features)),
    ]

    print(
        f"MLP benchmark: {args.samples} samples x {args.features} features, "
        f"{args.max_iter} Adam epochs (batch 200)"
    )

    # Resolve the mojo availability once so '+PLE' can mirror it.
    from shinrin._mlp._backend import get_mlp_backend

    try:
        os.environ.pop("SHINRIN_MLP_BACKEND", None)
    except Exception:
        pass
    resolved = get_mlp_backend()

    for kind, (X, y) in cases:
        print(f"\n=== {kind} ===")
        for backend in backends:
            ple = backend == "ple"
            if backend == "sklearn":
                bench_sklearn(kind, X, y, args.max_iter)
                continue
            if backend == "mojo":
                os.environ["SHINRIN_MLP_BACKEND"] = "mojo"
            elif backend == "numpy":
                os.environ["SHINRIN_MLP_BACKEND"] = "numpy"
            else:
                os.environ["SHINRIN_MLP_BACKEND"] = resolved
            try:
                bench_case(
                    f"shinrin {backend}", kind, X, y, args.max_iter, ple=ple
                )
            except Exception as exc:
                print(f"  shinrin {backend:15s} skipped ({type(exc).__name__}: {exc})")
            finally:
                os.environ["SHINRIN_MLP_BACKEND"] = resolved


if __name__ == "__main__":
    main()
