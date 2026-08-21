"""Benchmark the Rust and Mojo native backends against each other.

Usage:
    just bench-backends          # or: uv run python benchmarks/bench_backends.py

Measures fit / predict / partial_fit wall time for Mondrian trees and
forests under both backends on identical data and seeds, then prints a
comparison table. The Mojo shared library is built first if missing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

N_REPEATS = 5
DATASETS = [
    ("small", 1_000, 10),
    ("medium", 10_000, 20),
    ("large", 50_000, 50),
]


def load_backend(backend: str):

    old = os.environ.get("SHINRIN_BACKEND")
    os.environ["SHINRIN_BACKEND"] = backend
    try:
        import shinrin._backend as backend_mod

        backend_mod._CACHE.clear()
        return backend_mod.get_backend_module()
    finally:
        if old is None:
            os.environ.pop("SHINRIN_BACKEND", None)
        else:
            os.environ["SHINRIN_BACKEND"] = old


def make_data(n_samples: int, n_features: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.randn(n_samples) * 0.5).astype(np.float64)
    return X, y


def time_fit(mod, X, y, seed=3) -> float:
    tree = mod.Tree(X.shape[1], np.array([1], dtype=np.intp), 1)
    criterion = mod.MSE(1, X.shape[0])
    splitter = mod.MondrianSplitter(criterion, np.random.RandomState(seed))
    builder = mod.DepthFirstTreeBuilder(splitter, 2, 12)
    start = time.perf_counter()
    builder.build(tree, X, y.reshape(-1, 1))
    return time.perf_counter() - start


def time_predict(mod, X, y, seed=3) -> float:
    tree = mod.Tree(X.shape[1], np.array([1], dtype=np.intp), 1)
    criterion = mod.MSE(1, X.shape[0])
    splitter = mod.MondrianSplitter(criterion, np.random.RandomState(seed))
    builder = mod.DepthFirstTreeBuilder(splitter, 2, 12)
    builder.build(tree, X, y.reshape(-1, 1))

    start = time.perf_counter()
    for _ in range(N_REPEATS):
        tree.predict(X)
    return (time.perf_counter() - start) / N_REPEATS


def time_partial_fit(mod, X, y, n_chunks=4, seed=7) -> float:
    chunk = X.shape[0] // n_chunks
    start = time.perf_counter()
    tree = mod.Tree(X.shape[1], np.array([1], dtype=np.intp), 1)
    builder = mod.PartialFitTreeBuilder(2, 20, np.random.RandomState(seed))
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        builder.build(tree, X[sl], y[sl].reshape(-1, 1))
    return time.perf_counter() - start


def bench_case(name: str, fn, *args) -> dict[str, float]:
    times = {}
    for backend in ("rust", "mojo"):
        mod = load_backend(backend)
        best = min(fn(mod, *args) for _ in range(N_REPEATS))
        times[backend] = best
    ratio = times["rust"] / times["mojo"]
    print(
        f"  {name:<28} rust={times['rust'] * 1000:9.2f}ms  "
        f"mojo={times['mojo'] * 1000:9.2f}ms  "
        f"speedup={ratio:6.2f}x",
        flush=True,
    )
    return times


def main() -> None:
    so_path = REPO_ROOT / "src" / "shinrin" / "_native_mojo_core.so"
    if not so_path.exists():
        print("Mojo shared library missing; building...", flush=True)
        import subprocess

        subprocess.run(
            [
                "uv",
                "run",
                "mojo",
                "build",
                "src/shinrin/_native_mojo.mojo",
                "--emit",
                "shared-lib",
                "-o",
                "src/shinrin/_native_mojo_core.so",
            ],
            check=True,
            cwd=REPO_ROOT,
        )

    try:
        load_backend("mojo")
    except ImportError:
        print("The 'mojo' package is not installed; cannot benchmark.", flush=True)
        sys.exit(1)

    print(f"Benchmarking backends over {N_REPEATS} repeats (best of):\n", flush=True)
    header = f"  {'case':<28} {'rust':>12}  {'mojo':>12}  {'mojo vs rust':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, n_samples, n_features in DATASETS:
        X, y = make_data(n_samples, n_features)
        print(f"\n{name}: n_samples={n_samples}, n_features={n_features}", flush=True)
        bench_case(f"fit tree ({n_samples:,})", time_fit, X, y)
        bench_case(f"predict x{N_REPEATS} ({n_samples:,})", time_predict, X, y)
        bench_case(f"partial_fit 4 chunks ({n_samples:,})", time_partial_fit, X, y)

    print("\nNote: speedup > 1.0 means Mojo is faster than Rust.", flush=True)


if __name__ == "__main__":
    main()
