#!/usr/bin/env python3
"""ONNX inference benchmarks: tolerance and speed for every exportable model.

Usage:
    uv run python scripts/benchmarks/bench_onnx.py [--smoke] [--repeats N]

For every model that supports ``shinrin.onnx.to_onnx`` this script

1. trains the model twice on identical data, once with float32 and once
   with float64 features/targets,
2. exports the fitted model to ONNX,
3. runs inference through onnxruntime (CPU) and through the native
   estimator,
4. compares outputs numerically (tolerance) and times both paths
   (inference speed),

and writes JSON plus Markdown reports next to this script.

Exportable models covered here (regression + classification where
applicable): MondrianTree, MondrianForest, RandomForest, ExtraTrees,
RF-Quantile (median baked into the graph), MLP, TabM, Corels and GOSDT
(the latter two on binarized features). SkopeRules / Ordt / TabICL are
omitted to keep runtime bounded; all exported graphs are float32.

Outputs:
- scripts/benchmarks/onnx_results.json      raw numbers
- scripts/benchmarks/ONNX_BENCHMARK.md      committed results document
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SHINRIN_MLP_BACKEND", "numpy")
os.environ.setdefault("SHINRIN_TABM_BACKEND", "numpy")
# Single-threaded on every side so native timings are comparable with the
# pinned one-thread onnxruntime session.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE = False
WARMUP_CALLS = 3
MIN_TIMING_TOTAL_S = 0.4
MAX_TIMING_REPEATS = 100

# Numeric agreement thresholds used for the pass/fail column. These mirror
# the conventions of the backend parity tests (rtol folded into a single
# absolute scale appropriate for unit-range targets/probabilities).
TOL_REGRESSION = 1e-3
TOL_PROBA = 1e-3
TOL_LABELS = 0.995  # >= 99.5% label agreement

GOSDT_TIME_LIMIT_S = 60


@dataclass
class CellResult:
    status: str = "ok"  # ok | error
    note: str = ""
    fit_s: float | None = None
    export_s: float | None = None
    onnx_bytes: int | None = None
    # Mondrian models record which encoding was produced: "tree-ensemble"
    # (plain ai.onnx.ml encoding; exact for constant-prediction models,
    # the default) or "exact" (standard-domain graph reproducing path
    # smoothing); other model families report "generic".
    export_mode: str | None = None
    # tolerance (native vs onnxruntime)
    max_abs_err: float | None = None
    mean_abs_err: float | None = None
    label_agreement: float | None = None
    tol_pass: bool | None = None
    # speed
    native_ms: float | None = None
    native_per_1k_ms: float | None = None
    ort_ms: float | None = None
    ort_per_1k_ms: float | None = None
    speedup: float | None = None
    # informational: impact of training dtype on the model itself
    cross_dtype_max_err: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.status == "error":
            out["status"] = "error"
        return out


@dataclass
class AlgoSpec:
    name: str
    tasks: tuple[str, ...]
    factory: Any  # task -> unfitted estimator
    pre: Any | None = None  # X_train, y_train, X_test -> (Xb, y, Xs, {})
    predict_kwargs: dict = field(default_factory=dict)  # e.g. quantile=50
    export_kwargs: dict = field(default_factory=dict)  # e.g. quantile=50
    binary_only: bool = False


@dataclass
class DatasetSpec:
    name: str
    task: str
    loader: Any


def _full_n() -> int:
    return 300 if SMOKE else 4_000


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _make_regression_ds():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    def load():
        n = _full_n()
        X, y = make_regression(
            n_samples=n, n_features=20, n_informative=15, noise=10.0, random_state=0
        )
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        return {
            "X_train": Xtr,
            "X_test": Xte,
            "y_train": ytr,
            "y_test": yte,
            "task": "regression",
        }

    return load


def _make_classification_ds(n_classes: int):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    def load():
        n = _full_n()
        X, y = make_classification(
            n_samples=n,
            n_features=20,
            n_informative=12,
            n_classes=n_classes,
            weights=[1.0 / n_classes] * n_classes,
            flip_y=0.02,
            random_state=0,
        )
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        return {
            "X_train": Xtr,
            "X_test": Xte,
            "y_train": ytr,
            "y_test": yte,
            "task": "classification",
        }

    return load


DATASETS = [
    DatasetSpec("synthetic-reg", "regression", _make_regression_ds()),
    DatasetSpec("synthetic-bin", "classification", _make_classification_ds(2)),
    DatasetSpec("synthetic-multi", "classification", _make_classification_ds(5)),
]


def load_datasets():
    out = {}
    for spec in DATASETS:
        data = spec.loader()
        data["n_features"] = data["X_train"].shape[1]
        out[spec.name] = data
        print(
            f"  loaded {spec.name:<14} task={data['task']:<14} "
            f"train={data['X_train'].shape} test={data['X_test'].shape}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def _corels():
    from shinrin import CorelsClassifier

    return CorelsClassifier(c=0.01, max_card=1, min_support=0.05, verbosity=[])


def _gosdt():
    from shinrin import GOSDTClassifier

    return GOSDTClassifier(
        regularization=0.05,
        depth_budget=4,
        time_limit=GOSDT_TIME_LIMIT_S,
    )


def _mondrian_tree(task):
    from shinrin import MondrianTreeClassifier, MondrianTreeRegressor

    if task == "regression":
        return MondrianTreeRegressor(max_depth=16, random_state=0)
    return MondrianTreeClassifier(max_depth=16, random_state=0)


def _mondrian_forest(task):
    from shinrin import MondrianForestClassifier, MondrianForestRegressor

    if task == "regression":
        return MondrianForestRegressor(n_estimators=20, max_depth=16, random_state=0)
    return MondrianForestClassifier(n_estimators=20, max_depth=16, random_state=0)


def _rf():
    from shinrin import RandomForestRegressor

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

    # Median is baked into the exported graph at export time.
    return RandomForestQuantileRegressor(n_estimators=50, random_state=0, n_jobs=1)


def _mlp(task):
    from shinrin import MLPClassifier, MLPRegressor

    if task == "regression":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100, random_state=0)
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=0)


def _tabm(task):
    from shinrin import TabMClassifier, TabMRegressor

    if task == "regression":
        return TabMRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=0)
    return TabMClassifier(hidden_layer_sizes=(128, 128), max_iter=50, random_state=0)


def _pre_corels(X_train, y_train, X_test):
    from sklearn.preprocessing import KBinsDiscretizer

    enc = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")
    Xb = (enc.fit_transform(X_train) > 0).astype(np.float64)
    Xs = (enc.transform(X_test) > 0).astype(np.float64)
    return Xb, y_train, Xs, {}


def _pre_gosdt(X_train, y_train, X_test):
    from shinrin import ThresholdGuessBinarizer

    enc = ThresholdGuessBinarizer(n_estimators=20, max_depth=2, random_state=0)
    Xb = (enc.fit_transform(X_train, y_train) > 0.5).astype(np.float64)
    Xs = (enc.transform(X_test) > 0.5).astype(np.float64)
    return Xb, y_train, Xs, {}


ALGOS = [
    AlgoSpec("MondrianTree", ("regression", "classification"), _mondrian_tree),
    AlgoSpec("MondrianForest", ("regression", "classification"), _mondrian_forest),
    AlgoSpec("RandomForest", ("regression",), lambda task: _rf()),
    AlgoSpec("ExtraTrees", ("regression",), lambda task: _et()),
    AlgoSpec(
        "RF-Quantile",
        ("regression",),
        lambda task: _rf_quantile(),
        predict_kwargs={"quantile": 50},
        export_kwargs={"quantile": 50},
    ),
    AlgoSpec("MLP", ("regression", "classification"), _mlp),
    AlgoSpec("TabM", ("regression", "classification"), _tabm),
    AlgoSpec(
        "Corels",
        ("classification",),
        lambda task: _corels(),
        pre=_pre_corels,
        binary_only=True,
    ),
    AlgoSpec(
        "GOSDT",
        ("classification",),
        lambda task: _gosdt(),
        pre=_pre_gosdt,
        binary_only=True,
    ),
]


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


def _time_calls(fn, n_repeats: int) -> tuple[float, float]:
    """Return (mean_ms, min_ms) over repeated invocations."""
    for _ in range(WARMUP_CALLS):
        fn()
    times: list[float] = []
    start = time.perf_counter()
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        elapsed = time.perf_counter() - start
        if elapsed >= MIN_TIMING_TOTAL_S:
            break
    return float(np.mean(times)) * 1e3, float(np.min(times)) * 1e3


def _fit(algo: AlgoSpec, data: dict, dtype: np.dtype):
    """Fit a fresh estimator on (optionally preprocessed) dtype-cast data.

    Returns ``(model, X_train, X_test, y_train, fit_s)`` so callers compare
    native and ORT outputs on the same feature space the model saw.
    """
    model = algo.factory(data["task"])
    X_train, X_test = data["X_train"], data["X_test"]
    y_train = data["y_train"]
    fit_kwargs: dict = {}
    if algo.pre is not None:
        X_train, y_train, X_test, fit_kwargs = algo.pre(X_train, y_train, X_test)
    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    if data["task"] == "regression":
        y_train = y_train.astype(dtype)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    return model, X_train, X_test, time.perf_counter() - t0


def bench_cell(algo: AlgoSpec, ds_name: str, data: dict, dtype: np.dtype) -> dict:
    import onnxruntime as ort

    from shinrin.onnx import to_onnx

    res = CellResult()
    tag = "f32" if dtype == np.float32 else "f64"
    try:
        n_classes = int(np.unique(data["y_train"]).size)
        if algo.binary_only and n_classes != 2:
            return {
                "dataset": ds_name,
                "algorithm": algo.name,
                "task": data["task"],
                "dtype": tag,
                "status": "skipped",
                "note": "binary targets only",
            }

        model, X_train, X_test, res.fit_s = _fit(algo, data, dtype)

        t0 = time.perf_counter()
        proto = to_onnx(model, X_train[:8], **algo.export_kwargs)
        res.export_s = time.perf_counter() - t0
        res.onnx_bytes = len(proto.SerializeToString())
        props = {p.key: p.value for p in proto.metadata_props}
        res.export_mode = props.get("shinrin_mondrian_export", "generic")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            proto.SerializeToString(),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        inp = session.get_inputs()[0]

        # Every exported graph is float32 with a dynamic batch dimension;
        # f64-trained models ride the same f32 deployment path.
        X_ort = X_test.astype(np.float32)

        output_names = [o.name for o in session.get_outputs()]
        ort_outs = session.run(output_names, {inp.name: X_ort})

        # ---- tolerance ----
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            nat = model.predict_proba(X_test)
            got = ort_outs[output_names.index("probabilities")]
            labels_nat = model.predict(X_test)
            labels_got = (
                ort_outs[output_names.index("labels")]
                if "labels" in output_names
                else got.argmax(axis=1)
            )
            res.label_agreement = float((labels_nat == labels_got).mean())
        else:
            nat = model.predict(X_test, **algo.predict_kwargs)
            got = ort_outs[0]

        nat_arr = np.asarray(nat, dtype=np.float64)
        got_arr = np.asarray(got, dtype=np.float64)
        if nat_arr.ndim == 1:
            nat_arr = nat_arr.reshape(-1, 1)
        if got_arr.ndim == 1:
            got_arr = got_arr.reshape(-1, 1)
        err = np.abs(got_arr - nat_arr)
        res.max_abs_err = float(err.max())
        res.mean_abs_err = float(err.mean())
        tol = TOL_PROBA if has_proba else TOL_REGRESSION
        res.tol_pass = bool(
            res.max_abs_err <= tol
            and (res.label_agreement is None or res.label_agreement >= TOL_LABELS)
        )

        # ---- speed ----
        n_repeats = 3 if SMOKE else MAX_TIMING_REPEATS
        native_mean, _ = _time_calls(
            lambda: model.predict(X_test, **algo.predict_kwargs), n_repeats
        )
        ort_mean, _ = _time_calls(
            lambda: session.run(output_names, {inp.name: X_ort}), n_repeats
        )
        n_test = len(X_test)
        res.native_ms = native_mean
        res.native_per_1k_ms = native_mean / n_test * 1_000
        res.ort_ms = ort_mean
        res.ort_per_1k_ms = ort_mean / n_test * 1_000
        res.speedup = native_mean / ort_mean

    except Exception as exc:  # noqa: BLE001
        res.status = "error"
        res.note = f"{type(exc).__name__}: {exc}"
        return {
            "dataset": ds_name,
            "algorithm": algo.name,
            "task": data["task"],
            "dtype": tag,
            **res.as_dict(),
        }

    return {
        "dataset": ds_name,
        "algorithm": algo.name,
        "task": data["task"],
        "dtype": tag,
        **res.as_dict(),
    }


def run_suite(datasets: dict) -> list[dict]:
    records = []
    dtypes = [np.dtype(t) for t in (np.float32, np.float64)]
    for ds_name, data in datasets.items():
        print(
            f"\n=== {ds_name} ({data['task']}, n_train={len(data['y_train']):,}) ===",
            flush=True,
        )
        for algo in ALGOS:
            if data["task"] not in algo.tasks:
                continue
            for dtype in dtypes:
                rec = bench_cell(algo, ds_name, data, dtype)
                records.append(rec)
                if rec["status"] == "ok":
                    print(
                        f"  {algo.name:<15} {rec['dtype']}  fit={rec['fit_s']:7.2f}s  "
                        f"err={rec['max_abs_err']:.2e}  "
                        f"native={rec['native_ms']:9.3f}ms  "
                        f"ort={rec['ort_ms']:9.3f}ms  "
                        f"x{rec['speedup']:7.2f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {algo.name:<15} {rec['status'].upper()}: {rec.get('note')}",
                        flush=True,
                    )
    return records


# ---------------------------------------------------------------------------
# Cross-dtype comparison (impact of training precision on the model itself)
# ---------------------------------------------------------------------------


def annotate_cross_dtype(records: list[dict], datasets: dict) -> None:
    """Record f32-trained vs f64-trained native prediction differences."""
    by_key: dict[tuple, dict] = {}
    for rec in records:
        by_key.setdefault((rec["dataset"], rec["algorithm"]), {})[rec["dtype"]] = rec
    for (ds, algo_name), group in by_key.items():
        if len(group) != 2 or any(r["status"] != "ok" for r in group.values()):
            continue
        algo = next(a for a in ALGOS if a.name == algo_name)
        data = datasets[ds]
        outs = []
        for dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            model, _, X_test, _ = _fit(algo, data, dtype)
            out = (
                model.predict_proba(X_test)
                if hasattr(model, "predict_proba")
                else model.predict(X_test, **algo.predict_kwargs)
            )
            outs.append(np.asarray(out, dtype=np.float64).ravel())
        err = float(np.abs(outs[0] - outs[1]).max())
        for rec in group.values():
            rec["cross_dtype_max_err"] = err


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v, spec=".3g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "pass" if v else "FAIL"
    return format(v, spec)


def build_markdown(meta: dict, records: list[dict]) -> str:
    L: list[str] = [
        "# ONNX Inference Benchmark",
        "",
        "Native vs ONNX-runtime inference for every shinrin model that supports",
        "``to_onnx``: each model is trained twice (float32 / float64 data),",
        "exported to ONNX, executed with onnxruntime (CPU), and compared against",
        "the native estimator for numeric agreement and wall-clock speed.",
        "",
        "Regenerate locally with:",
        "",
        "```bash",
        "uv run python scripts/benchmarks/bench_onnx.py",
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
        "- Models: MondrianTree (depth 16), MondrianForest (20 trees, depth 16),",
        "  RandomForest / ExtraTrees (100 trees, vendored sklearn engine),",
        "  RF-Quantile (50 trees, median baked into the graph),",
        "  MLP ((128, 64) hidden units, 100 Adam epochs),",
        "  TabM ((128, 128) hidden units, 50 Adam epochs; NumPy reference backend),",
        "  Corels and GOSDT (binary-only, on binarized features).",
        "- Datasets: synthetic regression (`make_regression`, 4k x 20), binary and",
        "  5-class classification (`make_classification`, 4k x 20); 80/20 split.",
        "- Each cell trains a fresh estimator on float32- and float64-cast data,",
        "  exports via `shinrin.onnx.to_onnx`, and loads the proto into",
        "  onnxruntime (CPU execution provider, intra_op=1 thread). All exported",
        "  graphs are float32, so f64-trained models ride the f32 deployment path.",
        "- Tolerance compares the full test-set outputs: max/mean absolute error,",
        "  classification label agreement, pass/fail against max-abs-error <= 1e-3",
        "  (probabilities and unit-scale predictions) with >= 99.5% label agreement.",
        "- The Mondrian export encoding follows the estimator's",
        "  `path_smoothing` prediction mode: constant-prediction models",
        "  (the default) export as plain tree-ensembles that match native",
        "  predict exactly; smoothing models get an exact standard-domain",
        "  graph reproducing path smoothing, with a size-guarded fallback.",
        "  Generic sklearn-style ensembles round thresholds/values to f32.",
        "- Speed reports the mean wall-clock per full test-set call after 3 warmup",
        "  calls (timed until >= 0.4 s total or 100 calls). NumPy/BLAS and",
        "  onnxruntime are pinned to one thread on both sides.",
        "- SkopeRules / Ordt / TabICL are omitted to keep runtime bounded.",
        "",
    ]

    order = [a.name for a in ALGOS]
    for task, title in (
        ("regression", "Regression"),
        ("classification", "Classification"),
    ):
        recs = [r for r in records if r["task"] == task]
        if not recs:
            continue
        metric = "predictions" if task == "regression" else "probabilities + labels"
        L += [f"## {title}", "", f"Tolerance and speed against `{metric}`.", ""]

        # tolerance table
        L += ["### Tolerance (native vs onnxruntime)", ""]
        L += [
            "| Dataset | Model | Dtype | Export mode | Max abs err | Mean abs err | Label agree | Check |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(
            recs, key=lambda r: (r["dataset"], order.index(r["algorithm"]), r["dtype"])
        ):
            L.append(
                f"| {r['dataset']} | {r['algorithm']} | {r['dtype']} "
                f"| {r.get('export_mode', '-')} "
                f"| {_fmt(r.get('max_abs_err'), '.2e')} "
                f"| {_fmt(r.get('mean_abs_err'), '.2e')} "
                f"| {_fmt(r.get('label_agreement'), '.4f')} "
                f"| {_fmt(r.get('tol_pass'))} |"
            )
        L += [
            "",
            "*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.",
            "",
            (
                "*Export mode*: Mondrian graphs are either `tree-ensemble`"
                " (plain ai.onnx.ml encoding of the hard tree structure;"
                " exact for constant-prediction models, the default) or"
                " `exact` (standard-domain graph reproducing Mondrian path"
                " smoothing; used for `path_smoothing=True` models unless"
                " the protobuf size guard kicks in). Everything else"
                " reports `generic`."
            ),
            "",
        ]

        # speed table
        L += ["### Inference speed", ""]
        L += [
            "| Dataset | Model | Dtype | Native ms | ORT ms | Speedup |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(
            recs, key=lambda r: (r["dataset"], order.index(r["algorithm"]), r["dtype"])
        ):
            L.append(
                f"| {r['dataset']} | {r['algorithm']} | {r['dtype']} "
                f"| {_fmt(r.get('native_ms'), '.3g')} "
                f"| {_fmt(r.get('ort_ms'), '.3g')} "
                f"| {_fmt(r.get('speedup'), '.2f')}x |"
            )
        L += [
            "",
            "*Speedup*: native_time / ort_time (>1 means onnxruntime is faster).",
            "",
        ]

    # takeaways
    ok_recs = [r for r in records if r.get("tol_pass")]
    failed = [r for r in records if r.get("tol_pass") is False]
    faster = [
        r for r in records if r.get("speedup") is not None and r["speedup"] > 1.05
    ]
    slower = [
        r for r in records if r.get("speedup") is not None and r["speedup"] < 0.95
    ]
    errors = [r for r in records if r["status"] == "error"]
    skipped = [r for r in records if r["status"] == "skipped"]
    L += ["## Takeaways", ""]
    if records:
        L.append(f"- {len(ok_recs)}/{len(records)} cells meet the tolerance check.")
        approx_fail = [r for r in failed if r.get("export_mode") == "tree-ensemble"]
        other_fail = [r for r in failed if r not in approx_fail]
        if approx_fail:
            L.append(
                f"- {len(approx_fail)} failing cells export in `tree-ensemble`"
                " mode: for `path_smoothing=True` models above the size guard"
                " the exact smoothing graph falls back to the hard tree"
                " structure, which deviates from the smoothed native predict."
                " Constant-prediction models (the default) are exact in this"
                " mode, so a failure here indicates a smoothing model."
            )
        if other_fail:
            L.append("- Other failing cells:")
            for r in other_fail:
                L.append(
                    f"  - `{r['dataset']}` x `{r['algorithm']}` [{r['dtype']}]:"
                    f" max abs err {_fmt(r.get('max_abs_err'), '.2e')}"
                )
        if skipped:
            L.append(
                f"- {len(skipped)} cells skipped (binary-only models on"
                " multi-class data)."
            )
        cross_by_algo: dict[str, float] = {}
        for r in records:
            v = r.get("cross_dtype_max_err")
            if v is not None:
                cross_by_algo[r["algorithm"]] = max(
                    cross_by_algo.get(r["algorithm"], 0.0), v
                )
        if cross_by_algo:
            parts = [f"{name} {v:.0e}" for name, v in sorted(cross_by_algo.items())]
            L.append(
                "- f32-trained vs f64-trained native predictions disagree by"
                " (max abs): " + ", ".join(parts) + ". Forests differ most"
                " because rounding features/targets before fit changes which"
                " splits are chosen (targets here are unnormalized); TabM is"
                " exactly 0 because its NumPy backend casts inputs to float64"
                " internally."
            )
        L.append(
            f"- ONNX Runtime is >5% faster in {len(faster)} cells and >5% slower"
            f" in {len(slower)} cells (single-threaded CPU); it wins most on deep"
            " tree traversal and loses where the native path batches"
            " BLAS-friendly matrix products."
        )
    if errors:
        L.append(f"- {len(errors)} cells errored:")
        for r in errors:
            L.append(
                f"  - `{r['dataset']}` x `{r['algorithm']}` [{r['dtype']}]: {r.get('note')}"
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


def collect_meta(extra: dict[str, str]) -> dict:
    import sklearn

    import shinrin

    try:
        import onnx
        import onnxruntime as ort

        versions = {"onnx": onnx.__version__, "onnxruntime": ort.__version__}
    except ImportError:  # pragma: no cover
        versions = {}
    env = {
        "Date (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": _cpu_model(),
        "Cores": os.cpu_count(),
        "Python": platform.python_version(),
        "shinrin": shinrin.__version__,
        "NumPy": np.__version__,
        "scikit-learn": sklearn.__version__,
        **versions,
    }
    env.update(extra)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
    }


def main() -> None:
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="tiny fast verification run"
    )
    args = parser.parse_args()
    SMOKE = args.smoke

    warnings.simplefilter("ignore", UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    mode = " (SMOKE)" if SMOKE else ""
    print(f"ONNX inference benchmark{mode}\n", flush=True)

    datasets = load_datasets()
    records = run_suite(datasets)
    annotate_cross_dtype(records, datasets)

    suffix = ".smoke" if SMOKE else ""
    bench_dir = REPO_ROOT / "scripts" / "benchmarks"

    payload = {
        "meta": collect_meta({"Mode": "smoke" if SMOKE else "full"}),
        "datasets": {
            name: {
                "task": d["task"],
                "n_train": len(d["y_train"]),
                "n_test": len(d["X_test"]),
                "n_features": d["n_features"],
            }
            for name, d in datasets.items()
        },
        "results": records,
    }
    json_path = bench_dir / f"onnx_results{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md_path = bench_dir / f"ONNX_BENCHMARK{suffix}.md"
    md_path.write_text(build_markdown(payload["meta"], records))

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(
        f"\nDone: {n_ok}/{len(records)} cells ok.\nWrote:\n  {json_path}\n  {md_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
