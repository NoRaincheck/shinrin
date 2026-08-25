#!/usr/bin/env python3
"""Minimal-flip feature tweaking across SPOT, SPOTSET and random forests.

Experiment accompanying ``RASHOMON_FLIP_BENCHMARK.md``. Variation on
Tolomei et al. (KDD 2017, arXiv:1706.06691): instead of computing a feature
tweak against a single reference model, ask for the minimal tweak that flips
*every* member of a model set:

- ``spot/ref``          – single optimal sparse tree, classical CF;
- ``spotset/ref``       – first tree of the SPOTSET Rashomon set;
- ``spotset/rashomon``  – every tree of the Rashomon set simultaneously;
- ``rf{k}/ref``         – one tree out of a k-tree random forest;
- ``rf{k}/rashomon``    – every tree of the forest simultaneously.

Hypotheses:
1. Rashomon members share most decision structure, so the robust
   ``spotset/rashomon`` tweak costs barely more than ``spotset/ref``;
2. decorrelated forests make the same all-trees query far harder - larger
   distances, more search nodes, frequent infeasibility.

SPOT/SPOTSET tweaks are measured in binarized space (Hamming); forest
tweaks in raw space (L1). Distances compare within-family only.

Samples are test rows whose reference model predicts the negative class;
the requested flip is towards class 1.

Usage:
    uv run python scripts/benchmarks/bench_rashomon_flip.py
        [--dataset breast-cancer|compas|both] [--samples 40]
        [--forest-sizes 4,16,64] [--bound-multiplier 0.1]
        [--regularization 0.01] [--depth-budget 3] [--max-nodes 500000]
        [--time-limit 10] [--max-train 400] [--seed 0] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from shinrin import (
    RashomonFlipSearch,
    SPOTClassifier,
    SPOTSETClassifier,
    ThresholdGuessBinarizer,
    summarize_flip_results,
)

FOREST_DEPTH_CAP = 8


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "breast-cancer":
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        return np.asarray(data.data, dtype=float), np.asarray(data.target)
    if name == "compas":
        from shinrin._corels import load_from_csv

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
        X, y, _, _ = load_from_csv(path)
        return np.asarray(X, dtype=float), np.asarray(y)
    raise ValueError(f"unknown dataset {name!r}")


def prepare(
    name: str, X: np.ndarray, y: np.ndarray, seed: int, max_train: int
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    X_tr, y_tr = X_tr[:max_train], y_tr[:max_train]

    prep: dict[str, Any] = {"name": name, "X_train_raw": X_tr}
    if name == "breast-cancer":
        t0 = time.perf_counter()
        binarizer = ThresholdGuessBinarizer(
            n_estimators=20, max_depth=2, random_state=seed
        ).fit(X_tr, y_tr)
        prep["binarize_s"] = time.perf_counter() - t0
        prep["column_origins"] = [j for j, _ in binarizer.thresholds_]
        prep["X_train"] = np.asarray(binarizer.transform(X_tr), dtype=float)
        prep["X_test_bin"] = np.asarray(binarizer.transform(X_te), dtype=float)
    else:
        prep["binarize_s"] = 0.0
        prep["column_origins"] = None
        prep["X_train"] = X_tr.astype(float)
        prep["X_test_bin"] = X_te.astype(float)
    prep["X_test_raw"] = X_te
    prep["y_train"] = y_tr
    return prep


def fit_models(prep: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    models: dict[str, Any] = {}
    spot = SPOTClassifier(
        regularization=max(args.regularization, 1e-3),
        depth_budget=args.depth_budget,
    ).fit(prep["X_train"], prep["y_train"])
    models["spot"] = spot

    spotset = SPOTSETClassifier(
        regularization=args.regularization,
        rashomon_bound_multiplier=args.bound_multiplier,
        depth_budget=args.depth_budget,
        worker_limit=0,
        time_limit=120,
    ).fit(prep["X_train"], prep["y_train"])
    models["spotset"] = spotset

    forests: dict[int, Any] = {}
    for k in args.forest_sizes:
        forests[k] = RandomForestClassifier(
            n_estimators=k,
            max_depth=FOREST_DEPTH_CAP,
            random_state=args.seed,
            n_jobs=-1,
        ).fit(prep["X_train_raw"], prep["y_train"])
    models["forests"] = forests
    return models


def negative_rows(estimator: Any, X: np.ndarray, n: int, rng) -> list[int]:
    """Test rows the estimator's reference prediction sends to class 0."""
    preds = RashomonFlipSearch(estimator)._view.predict_all(X)[0]
    negatives = np.flatnonzero(preds == 0)
    pick = rng.permutation(negatives)[:n]
    return sorted(int(i) for i in pick)


def original_features_touched(result, column_origins: list[int] | None) -> float:
    if not result.success or not result.changed_features:
        return 0.0
    if column_origins is None:
        return float(len(result.changed_features))
    return float(len({column_origins[c] for c in result.changed_features}))


def run_configuration(
    key: str,
    estimator: Any,
    X_space: np.ndarray,
    rows: list[int],
    column_origins: list[int] | None,
    scope: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    search = RashomonFlipSearch(estimator)
    t0 = time.perf_counter()
    results = search.search(
        X_space[rows],
        target=1,
        scope=scope,
        max_nodes=args.max_nodes,
        time_limit=args.time_limit if scope == "rashomon" else None,
    )
    wall = time.perf_counter() - t0
    summary = summarize_flip_results(results)
    summary.update(
        key=f"{key}/{scope}",
        wall_s=round(wall, 3),
        mean_original_features=float(
            np.mean([original_features_touched(r, column_origins) for r in results])
        ),
        mean_agreement_before=float(
            np.mean([r.n_models_agree_before / r.n_models_total for r in results])
        ),
    )
    return summary


def evaluate_dataset(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    raw_X, raw_y = load_dataset(name)
    prep = prepare(name, raw_X, raw_y, args.seed, args.max_train)
    models = fit_models(prep, args)

    largest = max(models["forests"])
    configs = [
        ("spot", models["spot"], prep["X_test_bin"], prep["column_origins"]),
        ("spotset", models["spotset"], prep["X_test_bin"], prep["column_origins"]),
        (
            f"rf{largest}",
            models["forests"][largest],
            prep["X_test_raw"],
            None,
        ),
    ]

    rows: list[dict[str, Any]] = [
        {
            "dataset": name,
            "n_trees": 1 if key == "spot" else (
                models["spotset"].n_trees_ if key == "spotset" else largest
            ),
        }
        for key, _, _, _ in configs
    ]
    print(f"\n=== {name}: spotset has {models['spotset'].n_trees_} trees ===")
    for key, est, X_space, origins in configs:
        picked = negative_rows(est, X_space, args.samples, rng)
        scopes = ["reference", "rashomon"]
        for scope in scopes:
            summary = run_configuration(
                key, est, X_space, picked, origins, scope, args
            )
            summary["dataset"] = name
            rows.append(summary)
            print(format_row(summary))
    return rows


def format_row(s: dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.0f}%"

    dist = s["mean_distance"]
    dist = "-" if dist is None else f"{dist:.3f}"
    feats = s["mean_changed_features"]
    feats = "-" if feats is None else f"{feats:.2f}"
    orig = s["mean_original_features"]
    orig = "-" if not s["success_rate"] else f"{orig:.2f}"
    return (
        f"{s['key']:<22} n={s['n_samples']:>3} ok={pct(s['success_rate']):>4} "
        f"infeas={pct(s['proven_infeasible_rate']):>4} "
        f"budget={pct(s['budget_exhausted_rate']):>4} d={dist:>6} "
        f"feat={feats:>5} orig={orig:>5} agree={s['mean_agreement_before']:.2f} "
        f"nodes={s['total_nodes']:>8} t={s['wall_s']:>7.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="both", choices=["breast-cancer", "compas", "both"]
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--forest-sizes", default="4,16,64")
    parser.add_argument("--bound-multiplier", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--depth-budget", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.samples = 6
        args.forest_sizes = "4"
        args.max_nodes = 20_000
        args.time_limit = 2.0
        args.max_train = 150

    args.forest_sizes = [int(k) for k in str(args.forest_sizes).split(",")]
    datasets = (
        ["breast-cancer", "compas"] if args.dataset == "both" else [args.dataset]
    )

    all_rows: list[dict[str, Any]] = []
    for name in datasets:
        all_rows.extend(evaluate_dataset(name, args))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "rashomon_flip_results.json")
    with open(out_path, "w") as fh:
        json.dump(all_rows, fh, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
