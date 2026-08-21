#!/usr/bin/env python3
"""Benchmark TabM training: NumPy reference vs Mojo kernels.

Times ``TabMRegressor`` / ``TabMClassifier`` fits on synthetic datasets
(continuous-only and mixed continuous/categorical) for both backends.
A PyTorch reference (upstream TabM) can be added with ``--with-torch``
when the optional benchmark dependencies are installed:

    uv sync --group tabm-bench

Usage:
    python scripts/benchmarks/bench_tabm.py [--max-iter N] [--with-torch]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from shinrin.tabm import TabMClassifier, TabMRegressor


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
    y = (logits + 0.5 * rng.randn(n_samples)).argsort().argsort() % 3
    return X, y.astype(np.int64)


def make_mixed_classification(n_samples: int, n_features: int, seed: int = 2):
    """Continuous + low-cardinality 'categorical' columns (openml-ish)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Round half the columns to a few levels so they are detected as
    # categorical by the preprocessing pipeline.
    n_cat = n_features // 2
    for j in range(n_cat):
        X[:, j] = np.round(X[:, j] * 2.0) / 2.0  # ~6 levels
    logits = (
        X[:, :n_cat].sum(axis=1)
        + 0.5 * X[:, n_cat:].sum(axis=1)
        + 0.3 * rng.randn(n_samples)
    )
    y = (logits > np.median(logits)).astype(np.int64)
    return X, y


def bench_backend(backend: str, max_iter: int, n_samples: int, n_features: int):
    print(f"\n--- backend: {backend} ---")
    cases = [
        ("regression", make_regression(n_samples, n_features)),
        ("binary+multiclass", make_classification(n_samples, n_features)),
        ("mixed categorical", make_mixed_classification(n_samples, n_features)),
    ]
    for name, (X, y) in cases:
        if name == "regression":
            model = TabMRegressor(
                max_iter=max_iter, random_state=0, solver="adam"
            )
        else:
            model = TabMClassifier(
                max_iter=max_iter, random_state=0, solver="adam"
            )
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
        score = model.score(X, y)
        print(
            f"  {name:20s} fit {elapsed:7.2f}s  "
            f"score {score:.4f}  ({n_samples}x{X.shape[1]}, {max_iter} epochs)"
        )


def bench_torch(max_iter: int, n_samples: int, n_features: int):
    """Reference timings with the upstream PyTorch TabM (optional).

    Targets ``yandex-research/tabm``'s ``tabm_reference.Model``; the exact
    upstream API moves between releases, so any mismatch is reported and
    skipped rather than failing the benchmark run.
    """
    try:
        import torch
        from tabm_reference import Model
    except ImportError as exc:
        print(f"\n--- torch reference unavailable ({exc}) ---")
        return
    print("\n--- backend: torch (upstream reference) ---")
    del n_features  # the reference builds its own architecture defaults
    for name, (X, y) in [
        ("regression", make_regression(n_samples, n_features)),
        ("mixed categorical", make_mixed_classification(n_samples, n_features)),
    ]:
        try:
            xt = torch.tensor(X)
            yt = torch.tensor(
                y, dtype=torch.float32 if name == "regression" else torch.long
            )
            d_out = 1 if name == "regression" else 2
            if hasattr(Model, "make_kan_based"):
                model = Model.make_kan_based(xt.shape[1], d_out)
            elif hasattr(Model, "make_plr_embeddings"):
                model = Model.make_plr_embeddings(d_out, 8, xt.shape[1])
            else:
                model = Model(xt.shape[1], d_out)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            t0 = time.perf_counter()
            model.train()
            for _ in range(max_iter):
                opt.zero_grad()
                out = model(xt, None)
                loss = out.loss.mean() if hasattr(out, "loss") else out.mean()
                loss.backward()
                opt.step()
            elapsed = time.perf_counter() - t0
            print(f"  {name:20s} fit {elapsed:7.2f}s  ({n_samples}x{X.shape[1]})")
        except Exception as exc:
            print(f"  {name:20s} skipped ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument(
        "--backends", default="numpy,mojo", help="comma list: numpy,mojo"
    )
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="also time the upstream PyTorch reference (needs tabm-bench deps)",
    )
    args = parser.parse_args()

    print(
        f"TabM benchmark: {args.samples} samples x {args.features} features, "
        f"{args.max_iter} Adam epochs"
    )
    for backend in args.backends.split(","):
        backend = backend.strip()
        if not backend:
            continue
        try:
            os.environ["SHINRIN_TABM_BACKEND"] = backend
            bench_backend(backend, args.max_iter, args.samples, args.features)
        except Exception as exc:
            print(f"\n--- backend: {backend} unavailable ({exc}) ---")
    if args.with_torch:
        bench_torch(args.max_iter, args.samples, args.features)


if __name__ == "__main__":
    main()
