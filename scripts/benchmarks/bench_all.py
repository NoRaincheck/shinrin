#!/usr/bin/env python3
"""Benchmark every runnable shinrin algorithm across a suite of datasets.

Usage:
    uv run python scripts/benchmarks/bench_all.py [--smoke]

Measures fit time, predict time and test score (R^2 / accuracy) plus model
size for all shinrin estimators whose dependencies are available:

- Mondrian trees & forests (regression + classification, Rust backend)
- Random Forest / Extra Trees regressors (sklearn engine)
- Random Forest / Extra Trees quantile regressors
- MLP regressor & classifier (NumPy reference backend)
- TabM regressor & classifier (NumPy reference backend)
- GOSDT optimal sparse trees (threshold-guessing pipeline)
- CORELS optimal rule lists (one-hot binarized pipeline)

Skipped automatically when their optional dependencies are missing:
SkopeRules (pandas), TabICL (torch + checkpoint download).

Outputs:
- scripts/benchmarks/ALL_MODELS_BENCHMARK.md   committed results document
- docs/features/benchmark-results.md           published documentation page
- scripts/benchmarks/all_models_results.json   raw numbers

Run with --smoke first to verify the harness end to end on small data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SHINRIN_MLP_BACKEND", "numpy")
os.environ.setdefault("SHINRIN_TABM_BACKEND", "numpy")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from shinrin.benchmark import benchmark_model_size, benchmark_prediction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE = False
FAST_REPEATS = 3  # best-of-N fits for datasets up to FAST_ROW_LIMIT rows
FAST_ROW_LIMIT = 5_000
PREDICT_REPEATS = 10
GOSDT_TIME_LIMIT_S = 60

SMOKE_DATASETS = ("diabetes", "breast-cancer")


@dataclass
class CellResult:
    status: str = "ok"  # ok | error | skipped
    note: str = ""
    fit_s: float | None = None
    fit_mean_s: float | None = None
    fit_repeats: int = 0
    predict_ms: float | None = None
    predict_per_1k_ms: float | None = None
    score: float | None = None
    n_nodes: int | None = None
    n_leaves: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# A preprocessor fits on the training fold once and transforms both folds.
# It returns transformed arrays plus extra kwargs forwarded to ``fit``.
PreFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, Any, np.ndarray, dict[str, Any]],
]


@dataclass
class AlgoSpec:
    name: str
    tasks: tuple[str, ...]  # subset of {"regression", "classification"}
    factory: Callable[[str], Any]  # task -> unfitted estimator
    binary_only: bool = False
    max_rows: int | None = None  # skip datasets above this training-row count
    pre: PreFn | None = None


@dataclass
class DatasetSpec:
    name: str
    task: str
    source: str
    loader: Callable[[], dict[str, Any]]
    meta_extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _split(X, y, seed: int = 0, *, stratify=None):
    test_size = 0.2 if len(y) >= 250 else 0.25
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )


def _ds_diabetes():
    from sklearn.datasets import load_diabetes

    d = load_diabetes()
    Xtr, Xte, ytr, yte = _split(d.data.astype(np.float32), d.target)
    return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}


def _ds_friedman1(n: int):
    def load():
        from sklearn.datasets import make_friedman1

        X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=0)
        Xtr, Xte, ytr, yte = _split(X.astype(np.float32), y)
        return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}

    return load


def _ds_california(cap: int):
    def load():
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing()
        rng = np.random.RandomState(0)
        idx = rng.choice(len(d.data), size=min(cap, len(d.data)), replace=False)
        Xtr, Xte, ytr, yte = _split(d.data[idx].astype(np.float32), d.target[idx])
        return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}

    return load


def _ds_make_regression(n: int, f: int, informative: int):
    def load():
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=n,
            n_features=f,
            n_informative=informative,
            noise=10.0,
            random_state=0,
        )
        Xtr, Xte, ytr, yte = _split(X.astype(np.float32), y)
        return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}

    return load


def _ds_real_clf(loader):
    def load():
        d = loader()
        Xtr, Xte, ytr, yte = _split(
            d.data.astype(np.float32), d.target, stratify=d.target
        )
        return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}

    return load


def _ds_make_classification(n: int, f: int, classes: int, informative: int):
    def load():
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n,
            n_features=f,
            n_informative=informative,
            n_redundant=max(0, min(f // 5, f - informative)),
            n_classes=classes,
            n_clusters_per_class=min(2, max(1, 12 // classes)),
            weights=[1.0 / classes] * classes,
            flip_y=0.02,
            random_state=0,
        )
        Xtr, Xte, ytr, yte = _split(X.astype(np.float32), y, stratify=y)
        return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}

    return load


def build_dataset_specs() -> list[DatasetSpec]:
    from sklearn.datasets import (
        fetch_california_housing,  # noqa: F401  (network check happens at load)
        load_breast_cancer,
        load_digits,
        load_wine,
    )

    specs = [
        DatasetSpec("diabetes", "regression", "sklearn (real)", _ds_diabetes),
        DatasetSpec("friedman1-2k", "regression", "synthetic", _ds_friedman1(2_000)),
        DatasetSpec("friedman1-10k", "regression", "synthetic", _ds_friedman1(10_000)),
        DatasetSpec(
            "california-10k", "regression", "sklearn (real)", _ds_california(10_000)
        ),
        DatasetSpec(
            "make-regression-5k",
            "regression",
            "synthetic",
            _ds_make_regression(5_000, 25, 15),
        ),
        DatasetSpec(
            "breast-cancer",
            "classification",
            "sklearn (real)",
            _ds_real_clf(load_breast_cancer),
        ),
        DatasetSpec(
            "wine", "classification", "sklearn (real)", _ds_real_clf(load_wine)
        ),
        DatasetSpec(
            "digits", "classification", "sklearn (real)", _ds_real_clf(load_digits)
        ),
        DatasetSpec(
            "synthetic-binary-5k",
            "classification",
            "synthetic",
            _ds_make_classification(5_000, 20, 2, 12),
        ),
        DatasetSpec(
            "synthetic-binary-20k",
            "classification",
            "synthetic",
            _ds_make_classification(20_000, 50, 2, 30),
        ),
        DatasetSpec(
            "synthetic-multiclass-5k",
            "classification",
            "synthetic",
            _ds_make_classification(5_000, 20, 5, 12),
        ),
    ]
    if SMOKE:
        return [s for s in specs if s.name in SMOKE_DATASETS]
    return specs


def load_all_datasets(specs: list[DatasetSpec]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        t0 = time.perf_counter()
        data = spec.loader()
        elapsed = time.perf_counter() - t0
        data["task"] = spec.task
        data["source"] = spec.source
        data["n_features"] = int(data["X_train"].shape[1])
        out[spec.name] = data
        print(
            f"  loaded {spec.name:<24} train={data['X_train'].shape} "
            f"test={data['X_test'].shape} ({elapsed:.2f}s)",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def _pre_gosdt(X_train, y_train, X_test):
    from shinrin import ThresholdGuessBinarizer

    enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
    Xb = (enc.fit_transform(X_train, y_train) > 0.5).astype(np.int64)
    Xs = (enc.transform(X_test) > 0.5).astype(np.int64)
    return Xb, y_train, Xs, {}


def _pre_corels(X_train, y_train, X_test):
    enc = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")
    Xb = (enc.fit_transform(X_train) > 0).astype(np.int64)
    Xs = (enc.transform(X_test) > 0).astype(np.int64)
    names = [f"x{i}" for i in range(Xb.shape[1])]
    return Xb, y_train, Xs, {"features": names}


def _mondrian_tree(task: str):
    from shinrin import MondrianTreeClassifier, MondrianTreeRegressor

    if task == "regression":
        return MondrianTreeRegressor(max_depth=16, random_state=0)
    return MondrianTreeClassifier(max_depth=16, random_state=0)


def _mondrian_forest(task: str):
    from shinrin import MondrianForestClassifier, MondrianForestRegressor

    if task == "regression":
        return MondrianForestRegressor(n_estimators=20, max_depth=16, random_state=0)
    return MondrianForestClassifier(n_estimators=20, max_depth=16, random_state=0)


def _rf():
    from shinrin import RandomForestRegressor

    # Overrides: the vendored forest still defaults to criterion="mse" and
    # max_features="auto", both removed in modern scikit-learn.
    return RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_features=1.0,
        random_state=0,
    )


def _et():
    from shinrin import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_features=1.0,
        random_state=0,
    )


def _rf_quantile():
    from shinrin import RandomForestQuantileRegressor

    return RandomForestQuantileRegressor(n_estimators=50, random_state=0, n_jobs=1)


def _et_quantile():
    from shinrin import ExtraTreesQuantileRegressor

    return ExtraTreesQuantileRegressor(n_estimators=50, random_state=0, n_jobs=1)


def _mlp(task: str):
    from shinrin import MLPClassifier, MLPRegressor

    if task == "regression":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100, random_state=0)
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=0)


def _tabm(task: str):
    from shinrin import TabMClassifier, TabMRegressor

    if task == "regression":
        return TabMRegressor(hidden_layer_sizes=(256, 256), max_iter=60, random_state=0)
    return TabMClassifier(hidden_layer_sizes=(256, 256), max_iter=60, random_state=0)


def _gosdt():
    from shinrin import SPOTClassifier

    return SPOTClassifier(
        regularization=0.05,
        depth_budget=4,
        time_limit=GOSDT_TIME_LIMIT_S,
    )


def _corels():
    from shinrin import CorelsClassifier

    return CorelsClassifier(c=0.01, max_card=1, min_support=0.05, verbosity=[])


try:
    import pandas  # noqa: F401  # ty: ignore[unresolved-import]

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch  # noqa: F401  # ty: ignore[unresolved-import]

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


ALGOS: list[AlgoSpec] = [
    AlgoSpec("MondrianTree", ("regression", "classification"), _mondrian_tree),
    AlgoSpec("MondrianForest", ("regression", "classification"), _mondrian_forest),
    AlgoSpec("RandomForest", ("regression",), lambda t: _rf()),
    AlgoSpec("ExtraTrees", ("regression",), lambda t: _et()),
    AlgoSpec("RF-Quantile", ("regression",), lambda t: _rf_quantile()),
    AlgoSpec("ET-Quantile", ("regression",), lambda t: _et_quantile()),
    AlgoSpec("MLP", ("regression", "classification"), _mlp),
    AlgoSpec("TabM", ("regression", "classification"), _tabm, max_rows=12_000),
    AlgoSpec(
        "GOSDT",
        ("classification",),
        lambda t: _gosdt(),
        binary_only=True,
        max_rows=6_000,
        pre=_pre_gosdt,
    ),
    AlgoSpec(
        "CORELS",
        ("classification",),
        lambda t: _corels(),
        binary_only=True,
        max_rows=6_000,
        pre=_pre_corels,
    ),
]


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


def bench_cell(algo: AlgoSpec, data: dict[str, Any]) -> CellResult:
    task = data["task"]
    res = CellResult()

    n_classes = int(np.unique(data["y_train"]).size)
    if algo.binary_only and n_classes != 2:
        res.status, res.note = "skipped", "binary targets only"
        return res
    if algo.max_rows is not None and len(data["y_train"]) > algo.max_rows:
        res.status, res.note = "skipped", f"row cap {algo.max_rows:,}"
        return res

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    fit_kwargs: dict[str, Any] = {}

    if algo.pre is not None:
        X_train, y_train, X_test, fit_kwargs = algo.pre(X_train, y_train, X_test)

    n_rows = len(y_train)
    n_fits = 1 if SMOKE or n_rows > FAST_ROW_LIMIT else FAST_REPEATS

    model = algo.factory(task)
    times: list[float] = []
    for i in range(n_fits):
        candidate = model if i == 0 else algo.factory(task)
        t0 = time.perf_counter()
        candidate.fit(X_train, y_train, **fit_kwargs)
        times.append(time.perf_counter() - t0)
        model = candidate
    res.fit_s = min(times)
    res.fit_mean_s = float(np.mean(times))
    res.fit_repeats = n_fits

    try:
        pstats = benchmark_prediction(
            {algo.name: model}, X_test, n_repeats=2 if SMOKE else PREDICT_REPEATS
        )
        res.predict_ms = pstats[algo.name]["mean_time"] * 1e3
        res.predict_per_1k_ms = res.predict_ms / max(1, len(X_test)) * 1_000
    except Exception as exc:  # noqa: BLE001
        res.note = f"predict failed: {type(exc).__name__}: {exc}"

    try:
        res.score = float(model.score(X_test, y_test))
    except Exception as exc:  # noqa: BLE001
        extra = f"score failed: {type(exc).__name__}: {exc}"
        res.note = f"{res.note}; {extra}" if res.note else extra

    try:
        s = benchmark_model_size({algo.name: model})[algo.name]
        if s["n_nodes"]:
            res.n_nodes = s["n_nodes"]
            res.n_leaves = s["n_leaves"]
    except Exception as exc:  # noqa: BLE001
        res.note = (
            f"{res.note}; size failed: {exc}" if res.note else f"size failed: {exc}"
        )

    return res


def run_suite(datasets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ds_name, data in datasets.items():
        print(
            f"\n=== {ds_name} ({data['task']}, n_train={len(data['y_train']):,}, "
            f"n_features={data['n_features']}) ===",
            flush=True,
        )
        for algo in ALGOS:
            if data["task"] not in algo.tasks:
                continue
            t0 = time.perf_counter()
            try:
                cell = bench_cell(algo, data)
            except Exception as exc:  # noqa: BLE001
                cell = CellResult(status="error", note=f"{type(exc).__name__}: {exc}")
            cell_s = time.perf_counter() - t0
            records.append(
                {
                    "dataset": ds_name,
                    "algorithm": algo.name,
                    "task": data["task"],
                    **cell.as_dict(),
                }
            )
            if cell.status == "ok":
                print(
                    f"  {algo.name:<16} fit={cell.fit_s:9.3f}s  "
                    f"predict={cell.predict_ms:9.3f}ms  "
                    f"score={cell.score:.4f}  ({cell_s:.1f}s)",
                    flush=True,
                )
            else:
                print(f"  {algo.name:<16} {cell.status}: {cell.note}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def _pivot(records: list[dict], ds_names: list[str], fmt_fn) -> list[str]:
    """Render a dataset x algorithm table using ``fmt_fn(record)``."""
    algos: list[str] = []
    for r in records:
        if (
            r["dataset"] in ds_names
            and r["status"] == "ok"
            and r["algorithm"] not in algos
        ):
            algos.append(r["algorithm"])
    lines = ["| Dataset | " + " | ".join(algos) + " |", "|---|" + "---|" * len(algos)]
    for ds in ds_names:
        row = {r["algorithm"]: r for r in records if r["dataset"] == ds}
        cells = [fmt_fn(row[a]) for a in algos]
        lines.append("| " + ds + " | " + " | ".join(cells) + " |")
    return lines


def build_markdown(meta: dict, datasets: dict, records: list, docs_page: bool) -> str:
    reg_ds = [n for n, d in datasets.items() if d["task"] == "regression"]
    clf_ds = [n for n, d in datasets.items() if d["task"] == "classification"]

    def ok(r):
        return r is not None and r.get("status") == "ok"

    L: list[str] = []

    if docs_page:
        L += [
            "# Benchmark Results",
            "",
            "Wall-clock performance of all runnable shinrin algorithms measured across",
            "a suite of synthetic and real datasets. Produced by",
            "`scripts/benchmarks/bench_all.py`; regenerate locally with:",
            "",
            "```bash",
            "uv run python scripts/benchmarks/bench_all.py",
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
            "## Methodology",
            "",
            "- Fit time: best of 3 fits on datasets up to 5,000 training rows, otherwise a single fit.",
            "- Predict time: mean over 10 predictions of the held-out test set; also normalized per 1,000 samples.",
            "- Score: R^2 on the test set for regression, accuracy for classification.",
            "- All algorithms run single-threaded on identical train/test splits (`random_state=0`, 80/20).",
            "- MLP and TabM use their pure-NumPy reference backends; Mondrian models use the Rust backend.",
            "- Model configurations: MondrianTree depth 16; MondrianForest 20 trees, depth 16; RandomForest / ExtraTrees 100 trees; quantile forests 50 trees; MLP (128, 64) hidden units, 100 Adam epochs; TabM (256, 256) hidden units, 60 Adam epochs.",
            "- Scores reflect these fixed budgets, not tuned optima: MLP trains for only 100 epochs and can underperform on unscaled targets (see california-10k).",
            "- GOSDT runs behind the threshold-guessing binarization pipeline (`depth_budget=4`, 60 s search limit), capped at 6,000 training rows.",
            "- CORELS runs on quantile one-hot binarized features (`max_card=1`), capped at 6,000 training rows.",
            "- TabM capped at 12,000 training rows (NumPy reference trainer cost).",
            "- Not included: SkopeRules (requires optional `pandas`), TabICL (requires torch plus a downloaded checkpoint). See the other benchmark documents for those comparisons.",
            "",
            "Times are seconds unless stated otherwise; predict columns are",
            "milliseconds per full test-set call / per 1,000 samples.",
            "",
        ]

    def section(title: str, task_ds: list[str]) -> None:
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
        score_name = "Accuracy" if title == "Classification" else "R^2"
        L += [f"*{score_name} on the held-out test set.*", ""]
        L += _pivot(recs, task_ds, lambda r: f"{r['score']:.4f}" if ok(r) else "-")
        L.append("")

    section("Regression", reg_ds)
    section("Classification", clf_ds)

    skipped = [r for r in records if r["status"] != "ok"]
    if skipped and docs_page:
        L += ["## Skipped / failed cells", ""]
        for r in skipped:
            L.append(
                f"- `{r['dataset']}` x `{r['algorithm']}`: {r['status']} ({r['note']})"
            )
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Metadata & entry point
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    if sys.platform == "darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return platform.machine()
    return platform.processor() or "unknown"


def collect_meta() -> dict[str, Any]:
    import sklearn

    import shinrin

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "Date (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": _cpu_model(),
            "Cores": os.cpu_count(),
            "Python": platform.python_version(),
            "shinrin": shinrin.__version__,
            "NumPy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "Backends": "Rust (Mondrian); NumPy reference (MLP, TabM)",
        },
    }


def main() -> None:
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="tiny fast verification run"
    )
    args = parser.parse_args()
    SMOKE = args.smoke

    for category in (UserWarning, RuntimeWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    mode = " (SMOKE)" if SMOKE else ""
    print(f"Benchmarking all shinrin algorithms{mode}\n", flush=True)

    print("Loading datasets...", flush=True)
    datasets = load_all_datasets(build_dataset_specs())

    t0 = time.perf_counter()
    records = run_suite(datasets)
    total_min = (time.perf_counter() - t0) / 60

    payload = {
        "meta": collect_meta(),
        "datasets": {
            name: {
                "task": d["task"],
                "source": d["source"],
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
    json_path = bench_dir / f"all_models_results{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md_path = bench_dir / f"ALL_MODELS_BENCHMARK{suffix}.md"
    md_path.write_text(build_markdown(payload["meta"], datasets, records, False))

    docs_dir = REPO_ROOT / "docs" / "features"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / f"benchmark-results{suffix}.md"
    docs_path.write_text(build_markdown(payload["meta"], datasets, records, True))

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(
        f"\nDone: {n_ok}/{len(records)} cells ok in {total_min:.1f} min.\n"
        f"Wrote:\n  {json_path}\n  {md_path}\n  {docs_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
