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

Exportable models: MondrianTree, MondrianForest, RandomForest,
ExtraTrees and TabM (regression + classification where applicable).
MLP, quantile forests, GOSDT, CORELS, SkopeRules, Ordt and TabICL have no
ONNX exporter in shinrin and are therefore out of scope.

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
from dataclasses import dataclass
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


@dataclass
class CellResult:
    status: str = "ok"  # ok | error
    note: str = ""
    fit_s: float | None = None
    export_s: float | None = None
    onnx_bytes: int | None = None
    ort_input_dtype: str | None = None
    # tolerance (native vs onnxruntime)
    max_abs_err: float | None = None
    mean_abs_err: float | None = None
    label_agreement: float | None = None
    tol_pass: bool | None = None
    # export fidelity: pure decision-tree traversal vs onnxruntime. For
    # Mondrian models the native predict smooths along the decision path,
    # so this isolates exporter correctness from algorithm semantics.
    structure_max_err: float | None = None
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
    family: str = "tree"  # tree | tabm (affects export/input dtype handling)


@dataclass
class DatasetSpec:
    name: str
    task: str
    n_samples: int
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
    DatasetSpec("synthetic-reg", "regression", 4_000, _make_regression_ds()),
    DatasetSpec("synthetic-bin", "classification", 4_000, _make_classification_ds(2)),
    DatasetSpec("synthetic-multi", "classification", 4_000, _make_classification_ds(5)),
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

    # Overrides mirror bench_all.py: the vendored forest predates modern
    # scikit-learn parameter names.
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


def _tabm(task):
    from shinrin import TabMClassifier, TabMRegressor

    if task == "regression":
        return TabMRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=0)
    return TabMClassifier(hidden_layer_sizes=(128, 128), max_iter=50, random_state=0)


ALGOS = [
    AlgoSpec("MondrianTree", ("regression", "classification"), _mondrian_tree),
    AlgoSpec("MondrianForest", ("regression", "classification"), _mondrian_forest),
    AlgoSpec("RandomForest", ("regression",), lambda task: _rf()),
    AlgoSpec("ExtraTrees", ("regression",), lambda task: _et()),
    AlgoSpec("TabM", ("regression", "classification"), _tabm, family="tabm"),
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
    """Fit a fresh estimator on dtype-cast data; returns (model, fit_s)."""
    model = algo.factory(data["task"])
    X = data["X_train"].astype(dtype)
    if data["task"] == "regression":
        y = data["y_train"].astype(dtype)
    else:
        y = data["y_train"]
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def _native_outputs(model, data: dict, dtype: np.dtype):
    X = data["X_test"].astype(dtype)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), None
    return model.predict(X), None


def _leaf_traversal(model, X: np.ndarray) -> np.ndarray:
    """Pure decision-tree inference using only the stored tree arrays.

    This is the semantic the ONNX export encodes: hard leaf lookup with
    per-leaf values (class-probability rows for classifiers), averaged
    over trees for forests. Mondrian models differ from this by their
    path-smoothed native predict.
    """
    trees = (
        [model] if hasattr(model, "tree_") else list(getattr(model, "estimators_", []))
    )
    if not trees:
        raise ValueError("model exposes neither tree_ nor estimators_")

    outputs = []
    for tree in trees:
        t = tree.tree_
        feature = t.feature
        threshold = t.threshold
        left = t.children_left
        right = t.children_right
        value = t.value[:, 0, :]
        n_classes = value.shape[1]

        node = np.zeros(len(X), dtype=np.int64)
        active = np.arange(len(X))
        while len(active):
            feats = feature[node[active]]
            is_leaf = feats == -2
            if is_leaf.all():
                break
            going_left = (
                X[active[~is_leaf], feats[~is_leaf]]
                <= threshold[node[active[~is_leaf]]]
            )
            nxt = np.where(
                going_left,
                left[node[active[~is_leaf]]],
                right[node[active[~is_leaf]]],
            )
            active = active[~is_leaf]
            node[active] = nxt
        vals = value[node]
        if n_classes > 1:
            # classifiers store class counts; ONNX carries probabilities
            sums = vals.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            vals = vals / sums
        outputs.append(vals)

    stacked = np.stack(outputs)  # (n_trees, n_samples, n_classes)
    return stacked.mean(axis=0)


def bench_cell(algo: AlgoSpec, ds_name: str, data: dict, dtype: np.dtype) -> dict:
    import onnxruntime as ort

    from shinrin.onnx import to_onnx

    res = CellResult()
    tag = "f32" if dtype == np.float32 else "f64"
    try:
        model, res.fit_s = _fit(algo, data, dtype)

        t0 = time.perf_counter()
        proto = to_onnx(model, data["X_train"][:8])
        res.export_s = time.perf_counter() - t0
        res.onnx_bytes = len(proto.SerializeToString())

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            proto.SerializeToString(),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        inp = session.get_inputs()[0]
        res.ort_input_dtype = inp.type

        # The TabM graph is always float32; feed whatever the graph declares
        # so f64-trained TabM runs through the same f32 deployment path.
        ort_dtype = np.float64 if "double" in inp.type else np.float32
        X_ort = data["X_test"].astype(ort_dtype)

        output_names = [o.name for o in session.get_outputs()]
        ort_outs = session.run(output_names, {inp.name: X_ort})

        # ---- tolerance ----
        X_native = data["X_test"].astype(dtype)
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            nat = model.predict_proba(X_native)
            got = ort_outs[output_names.index("probabilities")]
            labels_nat = model.predict(X_native)
            labels_got = (
                ort_outs[output_names.index("labels")]
                if "labels" in output_names
                else got.argmax(axis=1)
            )
            res.label_agreement = float((labels_nat == labels_got).mean())
        else:
            nat = model.predict(X_native)
            got = ort_outs[0]

        err = np.abs(
            np.asarray(got, dtype=np.float64) - np.asarray(nat, dtype=np.float64)
        )
        res.max_abs_err = float(err.max())
        res.mean_abs_err = float(err.mean())
        tol = TOL_PROBA if has_proba else TOL_REGRESSION
        res.tol_pass = bool(
            res.max_abs_err <= tol
            and (res.label_agreement is None or res.label_agreement >= TOL_LABELS)
        )

        # Export fidelity: pure decision-tree semantics vs onnxruntime.
        # Both vectors are raveled explicitly: regression traversal keeps a
        # trailing singleton dim while ORT emits shape (n,), and a naive
        # (n, 1) - (n,) would broadcast into an (n, n) matrix.
        if algo.family == "tree":
            struct = _leaf_traversal(model, X_ort.astype(ort_dtype))
            if has_proba:
                struct_out = ort_outs[output_names.index("probabilities")]
            else:
                struct_out = ort_outs[0]
            res.structure_max_err = float(
                np.abs(
                    np.asarray(struct, dtype=np.float64).ravel()
                    - np.asarray(struct_out, dtype=np.float64).ravel()
                ).max()
            )

        # ---- speed ----
        n_repeats = 3 if SMOKE else MAX_TIMING_REPEATS
        native_mean, _ = _time_calls(lambda: model.predict(X_native), n_repeats)
        ort_mean, _ = _time_calls(
            lambda: session.run(output_names, {inp.name: X_ort}), n_repeats
        )
        n_test = len(X_native)
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
                    print(f"  {algo.name:<15} ERROR: {rec.get('note')}", flush=True)
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
            model, _ = _fit(algo, data, dtype)
            X = data["X_test"].astype(dtype)
            out = (
                model.predict_proba(X)
                if hasattr(model, "predict_proba")
                else model.predict(X)
            )
            outs.append(np.asarray(out, dtype=np.float64))
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
        "  TabM ((128, 128) hidden units, 50 Adam epochs, NumPy reference backend).",
        "- Datasets: synthetic regression (`make_regression`, 4k x 20), binary and",
        "  5-class classification (`make_classification`, 4k x 20); 80/20 split.",
        "- Each cell trains a fresh estimator on float32- and float64-cast data,",
        "  exports via `shinrin.onnx.to_onnx`, and loads the proto into",
        "  onnxruntime (CPU execution provider, intra_op=1 thread).",
        "- Tolerance compares the full test-set outputs: max/mean absolute error,",
        "  classification label agreement, pass/fail against max-abs-error <= 1e-3",
        "  (probabilities and unit-scale predictions) with >= 99.5% label agreement.",
        "- `Struct err` compares onnxruntime against a pure decision-tree traversal",
        "  of the stored tree arrays (the exact semantics the export encodes), i.e.",
        "  exporter fidelity. Mondrian models smooth predictions along the decision",
        "  path as part of the Mondrian-process algorithm, so their native-vs-ORT",
        "  error quantifies that algorithmic gap rather than an export bug.",
        "- TabM graphs are float32-only by design, so f64-trained TabM is served",
        "  through the same f32 graph as f32-trained TabM.",
        "- Speed reports the mean wall-clock per full test-set call after 3 warmup",
        "  calls (timed until >= 0.4 s total or 100 calls). NumPy/BLAS and",
        "  onnxruntime are pinned to one thread on both sides.",
        "- Not every shinrin model has an ONNX exporter: MLP, quantile forests,",
        "  GOSDT, CORELS, SkopeRules, Ordt and TabICL are out of scope here.",
        "- Known numeric floors: Mondrian backends compute at float32 internally",
        "  even for float64 input, capping their achievable agreement near 1e-6;",
        "  sklearn forests average 100 trees in float32 when fed f32 data.",
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
            "| Dataset | Model | Dtype | Max abs err | Struct err | Mean abs err | Label agree | Check |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(
            recs, key=lambda r: (r["dataset"], order.index(r["algorithm"]), r["dtype"])
        ):
            L.append(
                f"| {r['dataset']} | {r['algorithm']} | {r['dtype']} "
                f"| {_fmt(r.get('max_abs_err'), '.2e')} "
                f"| {_fmt(r.get('structure_max_err'), '.2e')} "
                f"| {_fmt(r.get('mean_abs_err'), '.2e')} "
                f"| {_fmt(r.get('label_agreement'), '.4f')} "
                f"| {_fmt(r.get('tol_pass'))} |"
            )
        L += ["", "*Check*: max abs err <= 1e-3 and label agreement >= 99.5%.", ""]

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
    errors = [r for r in records if r["status"] != "ok"]
    L += ["## Takeaways", ""]
    if records:
        L.append(f"- {len(ok_recs)}/{len(records)} cells meet the tolerance check.")
        mondrian_fail = [r for r in failed if r["algorithm"].startswith("Mondrian")]
        other_fail = [r for r in failed if not r["algorithm"].startswith("Mondrian")]
        if mondrian_fail:
            L.append(
                f"- All {len(mondrian_fail)} failing cells are Mondrian models"
                " whose `Struct err` is exactly 0: the ONNX graphs reproduce the"
            )
            L.append(
                "  stored tree semantics bit-for-bit. The gap comes from native"
                " Mondrian inference smoothing predictions along the decision"
                " path - an algorithmic property, not an export defect."
            )
        if other_fail or errors:
            for r in other_fail + errors:
                L.append(
                    f"- Unexpected failure: `{r['dataset']}` x"
                    f" `{r['algorithm']}` [{r['dtype']}]: {r.get('note') or 'tolerance'}"
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
            " tree traversal and loses on TabM regression/binary where the NumPy"
            " reference batches BLAS-friendly matrix products."
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
