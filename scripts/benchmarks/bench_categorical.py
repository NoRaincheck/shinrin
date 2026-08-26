#!/usr/bin/env python3
"""Ablation benchmark for automatic categorical-feature awareness.

Compares, on datasets with informative integer-coded categorical columns:

- ``auto``   : default behaviour (detect + CatBoost-style target-statistic
  encoding for Mondrian; target-statistic threshold axes for SPOT/SPOTSET)
- ``onehot`` : XGBoost-style per-value indicator encoding (SPOT/SPOTSET only)
- ``none``   : categorical handling disabled (integer codes treated as
  numeric — the historical behaviour)

Saves results to ``categorical_results.json`` and a rendered
``CATEGORICAL_BENCHMARK.md`` next to this script.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------


def make_pure_categorical(n=4000, seed=0):
    """Signal lives exclusively in integer-coded categoricals.

    Deliberately uses NON-CONTIGUOUS category sets (odd codes and one
    middle value) so raw-code thresholds need several leaves while a
    single target-statistic split isolates them at once.
    """
    rng = np.random.default_rng(seed)
    d_cat = 4
    cat = rng.integers(0, 10, size=(n, d_cat)).astype(np.float32)
    y = ((cat[:, 0] % 2 == 1) | (cat[:, 1] == 3)).astype(int)
    return cat, y, [f"cat{i}" for i in range(d_cat)]


def make_mixed(n=4000, seed=1):
    """Half high-cardinality integer-coded categoricals, half continuous."""
    rng = np.random.default_rng(seed)
    d_cat, d_num = 3, 3
    cat = rng.integers(0, 10, size=(n, d_cat)).astype(np.float32)
    num = rng.normal(size=(n, d_num)).astype(np.float32)
    logit = (
        1.5 * (cat[:, 0] % 2)
        - 1.2 * (cat[:, 1] == 7)
        + 1.2 * num[:, 0]
        + rng.normal(size=n) * 0.3
    )
    y = (logit > np.median(logit)).astype(int)
    X = np.hstack([cat, num])
    names = [f"cat{i}" for i in range(d_cat)] + [f"num{i}" for i in range(d_num)]
    return X, y, names


def load_compas():
    """Real data: compas-two-years is fully integer/binary coded."""
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
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(int)
    return X, y, [f"x{i}" for i in range(X.shape[1])]


DATASETS = {
    "pure-categorical": make_pure_categorical,
    "mixed": make_mixed,
    "compas": lambda: load_compas(),
}


# ---------------------------------------------------------------------------
# Model runners. Each returns (accuracy, fit+predict seconds).
# ---------------------------------------------------------------------------


def _bin_kwargs(mode, encoding):
    """Keyword args for the binarizer per ablation arm."""
    if mode == "auto":
        return {}
    if mode == "none":
        return {"categorical_features": None}
    return {"categorical_encoding": encoding}


def run_mondrian(X_tr, y_tr, X_te, y_te, mode):
    from shinrin import MondrianForestClassifier

    kw = {} if mode == "auto" else {"categorical_features": None}
    clf = MondrianForestClassifier(n_estimators=50, random_state=42, **kw)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    dt = time.perf_counter() - t0
    return accuracy_score(y_te, pred), dt, X_tr.shape[1]


def run_spot(X_tr, y_tr, X_te, y_te, mode, encoding="target"):
    from shinrin._spot import SPOTClassifier, ThresholdGuessBinarizer

    if mode == "auto":
        bin_kw = {}
    else:
        bin_kw = {
            "categorical_features": None,
            "categorical_encoding": encoding,
        }
    enc = ThresholdGuessBinarizer(n_estimators=60, random_state=42, **bin_kw)
    t0 = time.perf_counter()
    Xb = enc.fit_transform(X_tr, y_tr)
    clf = SPOTClassifier(
        depth_budget=4, regularization=0.05, worker_limit=1, time_limit=120
    )
    clf.fit(Xb, y_tr)
    pred = clf.predict(enc.transform(X_te))
    dt = time.perf_counter() - t0
    return accuracy_score(y_te, pred), dt, X_tr.shape[1]


def run_spotset(X_tr, y_tr, X_te, y_te, mode, encoding="target"):
    from shinrin import SPOTSETClassifier, ThresholdGuessBinarizer

    if mode == "auto":
        bin_kw = {}
    else:
        bin_kw = {
            "categorical_features": None,
            "categorical_encoding": encoding,
        }
    enc = ThresholdGuessBinarizer(
        n_estimators=40, max_depth=2, random_state=42, **bin_kw
    )
    t0 = time.perf_counter()
    Xb = enc.fit_transform(X_tr, y_tr)
    clf = SPOTSETClassifier(
        depth_budget=4,
        regularization=0.05,
        rashomon_bound_multiplier=0.05,
        worker_limit=1,
        time_limit=120,
    )
    clf.fit(Xb, y_tr)
    pred = clf.predict(enc.transform(X_te))
    dt = time.perf_counter() - t0
    return accuracy_score(y_te, pred), dt, Xb.shape[1]


RUNNERS = {
    ("mondrian", "auto"): lambda *a: run_mondrian(*a, "auto"),
    ("mondrian", "none"): lambda *a: run_mondrian(*a, "none"),
    ("spot", "auto"): lambda *a: run_spot(*a, "auto"),
    ("spot", "none"): lambda *a: run_spot(*a, "none"),
    ("spot", "onehot"): lambda *a: run_spot(*a, "onehot"),
    ("spotset", "auto"): lambda *a: run_spotset(*a, "auto"),
    ("spotset", "none"): lambda *a: run_spotset(*a, "none"),
    ("spotset", "onehot"): lambda *a: run_spotset(*a, "onehot"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        help="Subset of datasets to run",
    )
    args = parser.parse_args()

    results = []
    for ds_name in args.datasets:
        X, y, _ = DATASETS[ds_name]()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        for (model, mode), runner in RUNNERS.items():
            label = f"{model}/{ds_name}/{mode}"
            try:
                acc, secs, n_cols = runner(X_tr, y_tr, X_te, y_te)
                results.append(
                    {
                        "dataset": ds_name,
                        "model": model,
                        "mode": mode,
                        "accuracy": round(float(acc), 4),
                        "seconds": round(secs, 2),
                        "derived_columns": int(n_cols),
                    }
                )
                print(f"{label:34s} acc={acc:.4f}  cols={n_cols}  ({secs:.1f}s)")
            except Exception as exc:  # noqa: BLE001 - keep ablation going
                results.append(
                    {
                        "dataset": ds_name,
                        "model": model,
                        "mode": mode,
                        "error": str(exc)[:200],
                    }
                )
                print(f"{label:34s} FAILED: {exc}")

    json_path = RESULTS_DIR / "categorical_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    md_path = RESULTS_DIR / "CATEGORICAL_BENCHMARK.md"
    md_path.write_text(render_markdown(results))
    print(f"\nwrote {json_path}\nwrote {md_path}")


def render_markdown(results):
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df["accuracy"] = df.get("accuracy")
    lines = [
        "# Categorical-awareness ablation",
        "",
        "Accuracy on held-out data. `auto` detects integer-coded",
        "categorical columns and applies CatBoost-style smoothed target-",
        "statistic encoding (Mondrian) or target-statistic threshold axes",
        "(SPOT/SPOTSET); `onehot` is the XGBoost-style indicator baseline;",
        "`none` treats every column numerically (historical behaviour).",
        "",
    ]
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        lines += [
            f"## {ds_name}",
            "",
            "| model | mode | accuracy | derived columns | seconds |",
            "|---|---|---|---|---|",
        ]
        for _, row in sub.iterrows():
            acc = row.get("accuracy")
            acc_s = f"{acc:.4f}" if isinstance(acc, float) else "failed"
            secs = row.get("seconds", "")
            secs_s = f"{secs:.1f}" if isinstance(secs, float) else ""
            cols = row.get("derived_columns", "")
            cols_s = str(int(cols)) if pd.notna(cols) else ""
            lines.append(
                f"| {row['model']} | {row['mode']} | {acc_s} | {cols_s} | {secs_s} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
