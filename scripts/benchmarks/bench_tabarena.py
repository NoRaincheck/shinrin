#!/usr/bin/env python3
"""Benchmark all runnable shinrin algorithms on the TabArena-v0.1 core subset.

Usage:
    uv run --extra pandas python scripts/benchmarks/bench_tabarena.py [--smoke]

TabArena (Erickson et al., NeurIPS 2025, https://arxiv.org/abs/2506.16791)
is a living benchmark for tabular ML built on 51 curated real-world
datasets shared as OpenML suite 457 ("tabarena-v0.1"). This script runs
the same algorithm matrix as ``bench_all.py`` over a fixed *core subset*
of 13 TabArena datasets chosen to span regression / binary / multiclass
classification and numeric-only vs categorical-heavy feature spaces while
staying below ~5k rows so the full matrix finishes quickly:

- regression: fish-toxicity, concrete-strength, insurance-expenses,
  airfoil-noise, fiat-500
- binary classification: blood-transfusion, pima-diabetes, credit-g,
  qsar-biodeg, seismic-bumps, churn
- multiclass classification: anneal, maternal-health, website-phishing

Datasets are fetched from OpenML by dataset id (cached under
``~/scikit_learn_data`` after the first download). Scores are NOT
comparable with the public tabarena.ai leaderboard: this script uses a
single fixed split and untuned default budgets to compare shinrin's own
algorithms on curated real-world data.

Requires scikit-learn and pandas for the OpenML frame loader.

Outputs:
- scripts/benchmarks/TABARENA_BENCHMARK.md   committed results document
- scripts/benchmarks/tabarena_results.json   raw numbers

Run with --smoke first to verify the harness end to end on small data.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("SHINRIN_MLP_BACKEND", "numpy")
os.environ.setdefault("SHINRIN_TABM_BACKEND", "numpy")

import numpy as np

# The benchmark engine (CellResult/AlgoSpec/DatasetSpec, ALGOS matrix,
# bench_cell, run_suite, formatting helpers) is shared with bench_all.py so
# numbers stay comparable across the two suites.
from bench_all import DatasetSpec, _fmt, _pivot, collect_meta, run_suite
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SMOKE = False
SMOKE_DATASETS = ("fish-toxicity", "pima-diabetes", "maternal-health")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# Core subset of TabArena-v0.1 (OpenML suite 457):
# (name, task, openml dataset id, problem type, total rows, % categorical).
CORE_SUBSET = (
    # Regression
    ("fish-toxicity", "regression", 46954, "regression", 908, 0.0),
    ("concrete-strength", "regression", 46917, "regression", 1030, 0.0),
    ("insurance-expenses", "regression", 46931, "regression", 1338, 42.9),
    ("airfoil-noise", "regression", 46904, "regression", 1503, 16.7),
    ("fiat-500", "regression", 46907, "regression", 1538, 12.5),
    # Binary classification
    ("blood-transfusion", "classification", 46913, "binary", 748, 20.0),
    ("pima-diabetes", "classification", 46921, "binary", 768, 11.1),
    ("credit-g", "classification", 46918, "binary", 1000, 66.7),
    ("qsar-biodeg", "classification", 46952, "binary", 1054, 14.3),
    ("seismic-bumps", "classification", 46956, "binary", 2584, 25.0),
    ("churn", "classification", 46915, "binary", 5000, 25.0),
    # Multiclass classification
    ("anneal", "classification", 46906, "multiclass", 898, 84.6),
    ("maternal-health", "classification", 46941, "multiclass", 1014, 14.3),
    ("website-phishing", "classification", 46963, "multiclass", 1353, 100.0),
)


def _encode_categorical(s, mapping: dict[str, float], unknown: float) -> np.ndarray:
    """Map categories to codes; unseen/missing values get ``unknown``."""
    codes = np.array(
        [mapping.get(v, unknown) for v in s.astype(str)],
        dtype=np.float32,
    )
    codes[np.asarray(s.isna())] = unknown
    return codes


def _encode_features(Xtr, Xte):
    """Ordinal-encode categoricals / median-impute numerics.

    Statistics are computed on the training fold only; test categories
    unseen during training map to a reserved trailing code.
    """
    import pandas as pd

    cols_tr, cols_te = [], []
    for col in Xtr.columns:
        s_tr, s_te = Xtr[col], Xte[col]
        if pd.api.types.is_numeric_dtype(s_tr):
            tr = np.asarray(s_tr.to_numpy(dtype=np.float64, na_value=np.nan))
            finite = np.isfinite(tr)
            med = float(np.median(tr[finite])) if finite.any() else 0.0
            tr = np.where(np.isnan(tr), med, tr)
            te = np.asarray(s_te.to_numpy(dtype=np.float64, na_value=np.nan))
            te = np.where(np.isnan(te), med, te)
        else:
            mapping = {
                cat: float(i)
                for i, cat in enumerate(pd.unique(s_tr.dropna().astype(str)))
            }
            unknown = float(len(mapping))
            tr = _encode_categorical(s_tr, mapping, unknown)
            te = _encode_categorical(s_te, mapping, unknown)
        cols_tr.append(tr.astype(np.float32))
        cols_te.append(te.astype(np.float32))
    if not cols_tr:
        return (
            np.empty((len(Xtr), 0), dtype=np.float32),
            np.empty((len(Xte), 0), dtype=np.float32),
        )
    return np.column_stack(cols_tr), np.column_stack(cols_te)


def _encode_target(ytr_raw, yte_raw, task: str):
    """Integer-code classification targets; pass regression targets through."""
    import pandas as pd

    if task == "regression":
        y_train = np.asarray(ytr_raw, dtype=np.float64)
        y_test = np.asarray(yte_raw, dtype=np.float64)
        if np.isnan(y_train).any() or np.isnan(y_test).any():
            raise ValueError("missing values in regression target")
        return y_train, y_test
    tr_str = pd.Series(ytr_raw).astype(str)
    te_str = pd.Series(yte_raw).astype(str)
    codes, uniques = pd.factorize(tr_str)
    y_test = te_str.map({u: i for i, u in enumerate(uniques)})
    if y_test.isna().any():
        missing = sorted(te_str[y_test.isna()].unique())
        raise ValueError(f"test targets unseen in training fold: {missing}")
    return codes.astype(np.int64), y_test.to_numpy(np.int64)


def _ds_tabarena(data_id: int, task: str):
    def load():
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        X_df, y_raw = bunch.data, bunch.target
        stratify = y_raw if task == "classification" else None
        # 80/20 like bench_all.py (every core dataset has >=700 rows).
        Xtr_df, Xte_df, ytr_raw, yte_raw = train_test_split(
            X_df, y_raw, test_size=0.2, random_state=0, stratify=stratify
        )
        X_train, X_test = _encode_features(Xtr_df, Xte_df)
        y_train, y_test = _encode_target(ytr_raw, yte_raw, task)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    return load


def build_dataset_specs() -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            name=name,
            task=task,
            source=f"TabArena-v0.1 (OpenML d={data_id})",
            loader=_ds_tabarena(data_id, task),
            meta_extra={"problem_type": problem_type},
        )
        for name, task, data_id, problem_type, _, _ in CORE_SUBSET
    ]
    if SMOKE:
        return [s for s in specs if s.name in SMOKE_DATASETS]
    return specs


def load_datasets(specs: list[DatasetSpec]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        t0 = time.perf_counter()
        data = spec.loader()
        elapsed = time.perf_counter() - t0
        data["task"] = spec.task
        data["source"] = spec.source
        data.update(spec.meta_extra)
        data["n_features"] = int(data["X_train"].shape[1])
        out[spec.name] = data
        print(
            f"  loaded {spec.name:<22} train={data['X_train'].shape} "
            f"test={data['X_test'].shape} ({elapsed:.2f}s)",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_markdown(meta: dict, datasets: dict, records: list) -> str:
    reg_ds = [n for n, d in datasets.items() if d["task"] == "regression"]
    clf_ds = [n for n, d in datasets.items() if d["task"] == "classification"]

    def ok(r):
        return r is not None and r.get("status") == "ok"

    L: list[str] = [
        "# TabArena Benchmark",
        "",
        "Wall-clock performance of the runnable shinrin algorithms on a core",
        "subset of [TabArena](https://arxiv.org/abs/2506.16791)-v0.1, the living",
        "tabular ML benchmark ([OpenML suite 457](https://www.openml.org/s/457)).",
        "The core subset spans regression, binary and multiclass classification",
        "as well as numeric-only and categorical-heavy feature spaces; every",
        "dataset has at most ~5k rows. Produced by",
        "`scripts/benchmarks/bench_tabarena.py`; regenerate locally with:",
        "",
        "```bash",
        "uv run --extra pandas python scripts/benchmarks/bench_tabarena.py",
        "```",
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    for k, v in meta["environment"].items():
        L.append(f"| {k} | {v} |")

    L += [
        "",
        "## Datasets",
        "",
        "| Dataset | Task | Type | Rows | Features | % categorical |",
        "|---|---|---|---|---|---|",
    ]
    for name, task, _data_id, problem_type, rows, pct_cat in CORE_SUBSET:
        data = datasets.get(name)
        n_features = data["n_features"] if data else "-"
        L.append(
            f"| {name} | {task} | {problem_type} | {rows:,} | {n_features} "
            f"| {pct_cat:.1f} |"
        )

    L += [
        "",
        "## Methodology",
        "",
        "- Datasets are fetched from OpenML by their TabArena dataset id (cached under `~/scikit_learn_data`); categorical features are ordinal-encoded and numeric features median-imputed using training-fold statistics only. Test categories unseen during training map to a reserved trailing code.",
        "- Single stratified 80/20 train/test split (`random_state=0`). TabArena itself uses 3 outer folds x 10 repeats with tuned models, so scores here are **not** comparable with the [public leaderboard](https://tabarena.ai).",
        "- Fit time: best of 3 fits (all core-subset datasets are below the 5,000-row best-of-N threshold).",
        "- Predict time: mean over 10 predictions of the held-out test set; also normalized per 1,000 samples.",
        "- Score: R^2 on the test set for regression, accuracy for classification.",
        "- All algorithms run single-threaded on identical splits. MLP and TabM use their pure-NumPy reference backends; Mondrian models use the Rust backend.",
        "- Model configurations match `bench_all.py`: MondrianTree depth 16; MondrianForest 20 trees, depth 16; RandomForest / ExtraTrees 100 trees; quantile forests 50 trees; MLP (128, 64) hidden units, 100 Adam epochs; TabM (256, 256) hidden units, 60 Adam epochs.",
        "- Scores reflect these fixed budgets, not tuned optima.",
        "- GOSDT runs behind the threshold-guessing binarization pipeline (`depth_budget=4`, 60 s search limit); CORELS on quantile one-hot binarized features (`max_card=1`); both binary-classification only.",
        "- Not included: SkopeRules (requires optional `pandas` at fit time), TabICL (requires torch plus a downloaded checkpoint).",
        "",
        "Times are seconds unless stated otherwise; predict columns are",
        "milliseconds per full test-set call / per 1,000 samples.",
        "",
    ]

    def section(title: str, task_ds: list[str], score_name: str) -> None:
        nonlocal L
        if not task_ds:
            return
        recs = [r for r in records if r["dataset"] in task_ds]
        L += [f"## {title}", "", "*Fit time (seconds).*", ""]
        L += _pivot(recs, task_ds, lambda r: _fmt(r.get("fit_s")) if ok(r) else "-")
        L += ["", "*Predict: ms per full test-set call / ms per 1k samples.*", ""]
        L += _pivot(
            recs,
            task_ds,
            lambda r: (
                f"{_fmt(r.get('predict_ms'))} / {_fmt(r.get('predict_per_1k_ms'))}"
                if ok(r)
                else "-"
            ),
        )
        L.append("")
        L += [f"*{score_name} on the held-out test set.*", ""]
        L += _pivot(recs, task_ds, lambda r: f"{r['score']:.4f}" if ok(r) else "-")
        L.append("")

    section("Regression", reg_ds, "R^2")
    section("Classification", clf_ds, "Accuracy")

    skipped = [r for r in records if r["status"] != "ok"]
    if skipped:
        L += ["## Skipped / failed cells", ""]
        for r in skipped:
            L.append(
                f"- `{r['dataset']}` x `{r['algorithm']}`: {r['status']} ({r['note']})"
            )
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global SMOKE
    import bench_all  # engine module: share its globals (SMOKE, repeats)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="tiny fast verification run"
    )
    args = parser.parse_args()
    SMOKE = args.smoke
    bench_all.SMOKE = args.smoke

    for category in (UserWarning, RuntimeWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)

    mode = " (SMOKE)" if SMOKE else ""
    print(f"Benchmarking shinrin on the TabArena-v0.1 core subset{mode}\n", flush=True)

    print("Loading datasets from OpenML...", flush=True)
    datasets = load_datasets(build_dataset_specs())

    t0 = time.perf_counter()
    records = run_suite(datasets)
    total_min = (time.perf_counter() - t0) / 60

    payload = {
        "meta": collect_meta(),
        "datasets": {
            name: {
                "task": d["task"],
                "source": d["source"],
                "problem_type": d.get("problem_type"),
                "n_train": len(d["y_train"]),
                "n_test": len(d["y_test"]),
                "n_features": d["n_features"],
            }
            for name, d in datasets.items()
        },
        "results": records,
    }

    # Smoke runs must never clobber the published full-run artifacts.
    suffix = ".smoke" if SMOKE else ""
    bench_dir = REPO_ROOT / "scripts" / "benchmarks"
    json_path = bench_dir / f"tabarena_results{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path = bench_dir / f"TABARENA_BENCHMARK{suffix}.md"
    md_path.write_text(build_markdown(payload["meta"], datasets, records))

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(
        f"\nDone: {n_ok}/{len(records)} cells ok in {total_min:.1f} min.\n"
        f"Wrote:\n  {json_path}\n  {md_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
