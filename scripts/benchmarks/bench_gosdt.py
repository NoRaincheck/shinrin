#!/usr/bin/env python3
"""Benchmark the vendored GOSDT classifier against scikit-learn's CART.

Compares the optimal sparse decision tree pipeline (ThresholdGuessBinarizer +
SPOTClassifier) against a typical decision tree classifier
(sklearn.tree.DecisionTreeClassifier) on:

- ``cart``        : DecisionTreeClassifier on raw features (unlimited depth)
- ``cart+d``      : DecisionTreeClassifier with max_depth set to GOSDT's
                    depth budget (matched-complexity reference)
- ``gosdt``       : full reference-ensemble pipeline (binarize + optimize);
                    binarization time is reported separately. Datasets that
                    are already binary (compas) skip the binarizer.

The binarizer uses n_estimators=20, max_depth=2 by default: upstream's
defaults (100 x depth 3) produce hundreds of threshold columns whose column
elimination refits and whose resulting search spaces explode.

Metrics: binarize/fit/predict wall-clock times, held-out accuracy, tree
size, and (GOSDT only) the certified optimality bounds.

Usage:
    python scripts/benchmarks/bench_gosdt.py [--repeats N]
        [--workers 1,2,4,8]

With ``--workers``, a parallel-scaling sweep is run instead: the GOSDT stage
is fitted once per listed ``worker_limit`` value on each dataset (binarizing
only once) and fit-time speedups are reported alongside the certified loss,
which must be identical across all worker counts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from shinrin import SPOTClassifier, ThresholdGuessBinarizer


def compas_data() -> tuple[np.ndarray, np.ndarray]:
    path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "tests",
            "data",
            "compas.csv",
        )
    )
    from shinrin._corels import load_from_csv

    X, y, _, _ = load_from_csv(path)
    return np.asarray(X), np.asarray(y)


def gosdt_tree_stats(clf: SPOTClassifier) -> tuple[int, int]:
    """Return (leaves, total nodes) of the first extracted GOSDT model."""
    model = json.loads(clf.result_.model)[0]

    def count(node: dict[str, Any]) -> tuple[int, int]:
        if "prediction" in node:
            return 1, 1
        left_l, left_n = count(node["true"])
        right_l, right_n = count(node["false"])
        return left_l + right_l, left_n + right_n + 1

    return count(model)


def bench_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    depth_budget: int,
    regularization: float,
    **kwargs,
) -> list[dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    rows: list[dict[str, Any]] = []

    # --- CART baselines ---------------------------------------------------
    for label, cart_args in (
        ("cart", {}),
        ("cart+d", {"max_depth": depth_budget}),
    ):
        clf = DecisionTreeClassifier(random_state=0, **cart_args)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        preds = clf.predict(X_test)
        pred_ms = (time.perf_counter() - t0) * 1e3
        rows.append(
            {
                "model": label,
                "binarize_s": None,
                "fit_s": fit_s,
                "pred_ms": pred_ms,
                "test_acc": accuracy_score(y_test, preds),
                "train_acc": accuracy_score(y_train, clf.predict(X_train)),
                "nodes": int(clf.tree_.node_count),
                "leaves": int(clf.get_n_leaves()),
                "gap": None,
            }
        )

    # --- GOSDT pipeline ----------------------------------------------------
    prebinarized = bool(kwargs.pop("prebinarized", False))
    if prebinarized:
        Xb_train_raw, Xb_test_raw = X_train.astype(np.uint8), X_test.astype(np.uint8)
        bin_s = None
    else:
        enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
        t0 = time.perf_counter()
        Xb_train_raw = enc.fit_transform(X_train, y_train)
        bin_s = time.perf_counter() - t0
        Xb_test_raw = enc.transform(X_test)
    Xb_train = np.asarray(Xb_train_raw) > 0.5
    Xb_test = np.asarray(Xb_test_raw) > 0.5

    clf = SPOTClassifier(regularization=regularization, depth_budget=depth_budget)
    t0 = time.perf_counter()
    clf.fit(np.asarray(Xb_train) > 0.5, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = clf.predict(np.asarray(Xb_test) > 0.5)
    pred_ms = (time.perf_counter() - t0) * 1e3

    leaves, nodes = gosdt_tree_stats(clf)
    result = clf.get_result()
    rows.append(
        {
            "model": "gosdt",
            "binarize_s": bin_s,
            "fit_s": fit_s,
            "pred_ms": pred_ms,
            "test_acc": accuracy_score(y_test, preds),
            "train_acc": accuracy_score(
                y_train, clf.predict(np.asarray(Xb_train) > 0.5)
            ),
            "nodes": nodes,
            "leaves": leaves,
            "gap": result["upper_bound"] - result["lower_bound"],
        }
    )
    _ = name
    return rows


def print_table(rows: list[dict[str, Any]], flush: bool = False) -> None:
    header = (
        f"{'model':<8} {'binarize s':>11} {'fit s':>9} {'pred ms':>9} "
        f"{'test acc':>9} {'nodes':>6} {'leaves':>7} {'cert gap':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        bin_s = "-" if row["binarize_s"] is None else f"{row['binarize_s']:.3f}"
        gap = "-" if row["gap"] is None else f"{row['gap']:.4f}"
        print(
            f"{row['model']:<8} {bin_s:>11} {row['fit_s']:>9.3f} {row['pred_ms']:>9.2f} "
            f"{row['test_acc']:>9.4f} {row['nodes']:>6} {row['leaves']:>7} {gap:>9}",
            flush=flush,
        )


def bench_worker_scaling(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    depth_budget: int,
    regularization: float,
    workers: list[int],
    **kwargs,
) -> list[dict[str, Any]]:
    """Fit the GOSDT stage once per worker count and report scaling.

    The binarizer (if any) runs exactly once per dataset so that only the
    search itself is timed across ``worker_limit`` values.
    """
    prebinarized = bool(kwargs.pop("prebinarized", False))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    if prebinarized:
        Xb_train_raw, Xb_test_raw = X_train.astype(np.uint8), X_test.astype(np.uint8)
    else:
        enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
        Xb_train_raw = enc.fit_transform(X_train, y_train)
        Xb_test_raw = enc.transform(X_test)
    Xb_train = np.asarray(Xb_train_raw) > 0.5
    Xb_test = np.asarray(Xb_test_raw) > 0.5

    rows: list[dict[str, Any]] = []
    for worker_limit in workers:
        clf = SPOTClassifier(
            regularization=regularization,
            depth_budget=depth_budget,
            worker_limit=worker_limit,
        )
        t0 = time.perf_counter()
        clf.fit(Xb_train, y_train)
        fit_s = time.perf_counter() - t0
        preds = clf.predict(Xb_test)
        leaves, nodes = gosdt_tree_stats(clf)
        result = clf.get_result()
        rows.append(
            {
                "dataset": name,
                "workers": worker_limit,
                "fit_s": fit_s,
                "test_acc": accuracy_score(y_test, preds),
                "nodes": nodes,
                "leaves": leaves,
                "loss": result["model_loss"],
                "lower_bound": result["lower_bound"],
                "upper_bound": result["upper_bound"],
                "gap": result["upper_bound"] - result["lower_bound"],
            }
        )
        print(
            f"  workers={worker_limit}: fit={fit_s:.3f}s loss={result['model_loss']:.6f} "
            f"bounds=[{result['lower_bound']:.6f}, {result['upper_bound']:.6f}] "
            f"acc={accuracy_score(y_test, preds):.4f}",
            flush=True,
        )

    # Certified-loss parity: parallel search must not change the optimum.
    losses = {row["loss"] for row in rows}
    bounds = {(row["lower_bound"], row["upper_bound"]) for row in rows}
    status = "OK" if len(losses) == 1 and len(bounds) == 1 else "MISMATCH"
    print(f"  parity: {status}", flush=True)
    return rows


def print_scaling_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'dataset':<38} {'workers':>7} {'fit s':>9} {'speedup':>8} "
        f"{'test acc':>9} {'leaves':>7} {'cert gap':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset_rows in by_dataset.values():
        base_s = dataset_rows[0]["fit_s"]
        for row in dataset_rows:
            speedup = base_s / row["fit_s"] if row["fit_s"] > 0 else float("nan")
            gap = "-" if row["gap"] is None else f"{row['gap']:.4f}"
            print(
                f"{row['dataset']:<38} {row['workers']:>7} {row['fit_s']:>9.3f} "
                f"{speedup:>7.2f}x {row['test_acc']:>9.4f} {row['leaves']:>7} {gap:>9}",
                flush=True,
            )


WORKLOADS = [
    ("iris (real, n=150, d=4, 3 classes)", "iris", {}),
    (
        "small (n=2000, d=10)",
        "synthetic",
        {"n_samples": 2000, "n_features": 10, "n_informative": 6, "random_state": 1},
    ),
    (
        "medium (n=10000, d=20)",
        "synthetic",
        {"n_samples": 10000, "n_features": 20, "n_informative": 10, "random_state": 2},
    ),
    # compas features are already binary: skip the binarizer stage
    ("compas (real, binary, n=7214, d=27)", "compas", {"prebinarized": True}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--depth-budget", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=str,
        default=None,
        help=(
            "Comma-separated worker_limit values for a parallel-scaling "
            "sweep (e.g. --workers 1,2,4,8). Runs the GOSDT stage only; "
            "CART baselines are skipped."
        ),
    )
    args = parser.parse_args()

    datasets: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for name, kind, kwargs in WORKLOADS:
        if kind == "iris":
            X, y = load_iris(return_X_y=True)
        elif kind == "compas":
            X, y = compas_data()
        else:
            X, y = make_classification(**kwargs)
        datasets.append((name, np.asarray(X), np.asarray(y), dict(kwargs)))

    if args.workers is not None:
        workers = [int(w) for w in args.workers.split(",") if w.strip()]
        print(
            f"GOSDT worker-scaling sweep (depth_budget={args.depth_budget}, "
            f"regularization={args.regularization}, workers={workers}, "
            "75/25 stratified split)"
        )
        rows: list[dict[str, Any]] = []
        for name, X, y, extra in datasets:
            print()
            print(f"### {name}", flush=True)
            rows.extend(
                bench_worker_scaling(
                    name, X, y, args.depth_budget, args.regularization, workers, **extra
                )
            )
        print_scaling_table(rows)
        return

    print(
        f"GOSDT vs CART benchmark (depth_budget={args.depth_budget}, "
        f"regularization={args.regularization}, 75/25 stratified split)"
    )
    for name, X, y, extra in datasets:
        print()
        print(f"### {name}", flush=True)
        rows = bench_dataset(
            name, X, y, args.depth_budget, args.regularization, **extra
        )
        print_table(rows, flush=True)

    print()
    print("notes: cart+d matches GOSDT's depth budget; 'cert gap' is the")
    print("certified upper_bound - lower_bound optimality interval (GOSDT only).")
    print("use --workers for the parallel-scaling sweep.")


if __name__ == "__main__":
    main()
