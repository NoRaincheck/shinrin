#!/usr/bin/env python3
"""Benchmark the vendored CORELS classifier fit.

Times ``CorelsClassifier.fit`` on the COMPAS dataset and synthetic wide
datasets, comparing the two compile-time bit-vector configurations:

- default      : ``-DGMP`` with bundled mini-gmp (no system dependency)
- SHINRIN_CORELS_NO_GMP=1 : CORELS' word-array fallback (upstream's
                            no-GMP configuration)

Rebuild between configurations before re-running:

    uv run maturin develop --release                          # GMP (default)
    SHINRIN_CORELS_NO_GMP=1 uv run maturin develop --release  # no GMP

Usage:
    python scripts/benchmarks/bench_corels.py [--repeats N]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from shinrin._native import corels_gmp_enabled

from shinrin import CorelsClassifier
from shinrin._corels import load_from_csv


def compas_data():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "tests",
        "data",
        "compas.csv",
    )
    return load_from_csv(os.path.normpath(path))


def make_synthetic(n_samples: int, n_features: int, seed: int = 0):
    """Wide-dataset workload: many samples stress the bit-vector width."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint8)
    # Correlated labels so the search does real work: majority vote of a
    # subset plus label noise.
    signal = X[:, : max(2, n_features // 4)].sum(axis=1)
    y = np.where(signal > signal.max() // 2, 1, 0)
    flip = rng.random(n_samples) < 0.05
    y[flip] ^= 1
    return np.ascontiguousarray(X), y.astype(np.uint8)


# (name, params, synthetic_shape or None for COMPAS)
WORKLOADS = [
    ("compas", {}, None),
    ("compas", {"c": 0.001, "n_iter": 100000}, None),
    ("compas", {"max_card": 1, "n_iter": 100000}, None),
    ("synthetic", {"c": 0.005, "n_iter": 20000}, (50000, 40)),
    ("synthetic", {"c": 0.002, "n_iter": 20000}, (200000, 24)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    mode = "gmp (bundled mini-gmp)" if corels_gmp_enabled() else "no-gmp (word arrays)"
    print(f"CORELS fit benchmark - {mode}")
    print(f"repeats per workload: {args.repeats} (best-of reported)")
    print()

    compas = compas_data()
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    header = (
        f"{'dataset':<18} {'params':<36} {'fit s':>8} {'rules':>6} {'train acc':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, params, shape in WORKLOADS:
        if shape is None:
            label = f"{name} (n={compas[0].shape[0]}, d={compas[0].shape[1]})"
            X, y, features = compas[0], compas[1], compas[2]
        else:
            if shape not in cache:
                cache[shape] = make_synthetic(*shape)
            X, y = cache[shape]
            features = [f"f{i}" for i in range(X.shape[1])]
            label = f"{name} {shape}"

        times = []
        clf = None
        for _ in range(args.repeats):
            clf = CorelsClassifier(verbosity=[], **params)
            t0 = time.perf_counter()
            clf.fit(X, y, features=features)
            times.append(time.perf_counter() - t0)

        best = min(times)
        n_rules = len(clf.rl().rules)
        acc = clf.score(X, y)
        param_str = " ".join(f"{k}={v}" for k, v in params.items()) or "defaults"
        print(f"{label:<18} {param_str:<36} {best:>8.3f} {n_rules:>6} {acc:>10.4f}")


if __name__ == "__main__":
    main()
