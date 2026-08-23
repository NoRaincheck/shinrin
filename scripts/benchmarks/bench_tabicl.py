#!/usr/bin/env python3
"""Benchmark TabICL inference across compute backends.

Times ``TabICLClassifier`` / ``TabICLRegressor`` fit (preprocessing +
optional KV-cache build) and predict (ensemble forward passes) separately
on synthetic datasets, for the ``numpy``, ``torch`` or ``mojo`` backend.
The upstream ``tabicl`` package can be compared with ``--with-upstream``
when the benchmark extra is installed:

    uv sync --extra tabicl-bench

Sections:

- estimator sweep (default): fit / predict / score over the size grid for
  every task, plus predict throughput (ms per 1k test rows).
- ``--quant-ablation``: fp vs ternary post-training quantization
  (per-row / per-tensor scales) on a fixed classification case — timing,
  held-out accuracy and effective zero fraction of the quantized weights.
- ``--cache-sweep``: predict time across ``batch_size x kv_cache`` combos
  on a fixed mid-size case, isolating the chunking cliff and the KV-cache
  fix.

All section results are merged per backend into a JSON artifact
(``scripts/benchmarks/tabicl_results.json`` by default); ``--smoke``
writes a suffixed file instead so committed results are not clobbered.

Examples:

    python scripts/benchmarks/bench_tabicl.py --quick --backend numpy
    python scripts/benchmarks/bench_tabicl.py --backend torch
    python scripts/benchmarks/bench_tabicl.py --backend mojo --quick --smoke
    python scripts/benchmarks/bench_tabicl.py --backend mojo \
        --quant-ablation --cache-sweep
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from shinrin.tabicl import (
    CLASSIFIER_CHECKPOINT,
    REGRESSOR_CHECKPOINT,
    TabICLClassifier,
    TabICLRegressor,
)

DEFAULT_JSON = Path(__file__).resolve().parent / "tabicl_results.json"

QUANT_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("fp", "none", ""),
    ("ternary/row", "ternary", "per_row"),
    ("ternary/tensor", "ternary", "per_tensor"),
)


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
    centers = np.linspace(-2.0, 2.0, 5)
    y = np.digitize(logits + 0.5 * rng.randn(n_samples), bins=centers)
    return X, y.astype(np.int64)


def make_mixed_classification(n_samples: int, n_features: int, seed: int = 2):
    """Continuous + low-cardinality 'categorical' columns."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    n_cat = max(1, n_features // 4)
    for j in range(n_cat):
        levels = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n_samples)
        X[:, j] = levels
    logits = (
        X[:, :n_cat].sum(axis=1)
        + 0.5 * X[:, n_cat:].sum(axis=1)
        + 0.3 * rng.randn(n_samples)
    )
    y = (logits > np.median(logits)).astype(np.int64)
    return X, y


def _timed(fn, repeats: int, warmup: bool = True):
    if warmup:
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return times.mean(), times.std()


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _env_meta(args: argparse.Namespace) -> dict[str, Any]:
    import importlib.util

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "shinrin": _module_version("shinrin"),
        "numpy": _module_version("numpy"),
        "torch": _module_version("torch"),
        "mojo_native": importlib.util.find_spec("shinrin._native_tabicl") is not None,
        "classifier_checkpoint": CLASSIFIER_CHECKPOINT,
        "regressor_checkpoint": REGRESSOR_CHECKPOINT,
        "n_estimators": args.n_estimators,
        "repeat": args.repeat,
    }


def save_results(
    path: Path, backend: str, meta: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Merge this run's records into the per-backend JSON artifact."""
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    data["schema"] = 1
    data.setdefault("runs", {})[backend] = {"meta": meta, "results": records}
    path.write_text(json.dumps(data, indent=2) + "\n")


def _base_record(
    model: Any, task: str, backend: str, X_train: np.ndarray, X_test: np.ndarray
) -> dict[str, Any]:
    return {
        "task": task,
        "backend": backend,
        "n_train": int(len(X_train)),
        "n_features": int(X_train.shape[1]),
        "n_test": int(len(X_test)),
        "kv_cache": bool(getattr(model, "kv_cache", False)),
        "batch_size": int(getattr(model, "batch_size", 0)),
        "quantization": str(getattr(model, "quantization", "none")),
        "quantization_granularity": str(getattr(model, "quantization_granularity", "")),
    }


def _fit_predict_record(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: str,
    backend: str,
    repeat: int,
    regression: bool,
    quiet_warnings: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = _base_record(model, task, backend, X_train, X_test)
    rec["status"] = "ok"
    if extra:
        rec.update(extra)

    def _run() -> None:
        fit_time, fit_std = _timed(lambda: model.fit(X_train, y_train), repeat)

        def predict():
            if regression:
                return model.predict(X_test)
            return model.predict_proba(X_test)

        predict_time, predict_std = _timed(predict, repeat)
        rec["fit_s"] = round(float(fit_time), 4)
        rec["fit_s_std"] = round(float(fit_std), 4)
        rec["predict_s"] = round(float(predict_time), 4)
        rec["predict_s_std"] = round(float(predict_std), 4)
        rec["predict_per_1k_ms"] = round(
            float(predict_time) / max(1, len(X_test)) * 1000.0, 2
        )
        rec["score"] = round(float(model.score(X_test, y_test)), 4)

    try:
        if quiet_warnings:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _run()
        else:
            _run()
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        rec["status"] = f"error: {type(exc).__name__}: {exc}"
        print(f"  {rec['label']:<28} FAILED: {rec['status']}")

    if rec["status"] == "ok":
        print(
            f"  {rec['label']:<28} "
            f"fit {rec['fit_s']:8.3f}s ±{rec['fit_s_std']:.3f}  "
            f"predict {rec['predict_s']:8.3f}s ±{rec['predict_s_std']:.3f}  "
            f"({rec['predict_per_1k_ms']:8.1f} ms/1k)  "
            f"score {rec['score']:.4f}"
        )
    return rec


def bench_case(
    task: str, n_samples: int, n_features: int, args: argparse.Namespace
) -> dict[str, Any]:
    backend = args.backend
    n_test = min(1000, max(200, n_samples // 10))
    if task == "regression":
        X, y = make_regression(n_samples + n_test, n_features)
    elif task == "classification":
        X, y = make_classification(n_samples + n_test, n_features)
    else:
        X, y = make_mixed_classification(n_samples + n_test, n_features)
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    common: dict[str, Any] = {
        "backend": backend,
        "n_estimators": args.n_estimators,
        "random_state": 42,
        "kv_cache": args.kv_cache,
        "batch_size": args.batch_size,
    }
    regression = task == "regression"
    model = TabICLRegressor(**common) if regression else TabICLClassifier(**common)

    label = f"{task} {n_samples}x{n_features}"
    return _fit_predict_record(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        task=task,
        backend=backend,
        repeat=args.repeat,
        regression=regression,
        extra={"label": label},
    )


def _ptq_sparsity(checkpoint_version: str, granularity: str) -> float:
    """Effective zero fraction over the tensors PTQ would ternarize."""
    from shinrin._quant import ternary_quantize_dequantize
    from shinrin._tabicl._checkpoint import ensure_npz
    from shinrin.tabicl import _ternary_post_training_quantize

    _, _, params = ensure_npz(
        filename=checkpoint_version, model_path=None, allow_auto_download=True
    )
    qparams = _ternary_post_training_quantize(params, granularity)
    total = zeros = 0
    for key, arr in params.items():
        q = qparams[key]
        if q is arr:
            continue  # tensor was left full precision
        qa = ternary_quantize_dequantize(np.asarray(arr, np.float32), granularity)
        total += int(qa.size)
        zeros += int((qa == 0).sum())
    return zeros / max(total, 1)


def bench_quant_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    """fp vs ternary PTQ: timing + held-out accuracy on a fixed case."""
    from sklearn.datasets import make_classification as sk_make

    backend = args.backend
    n_train, n_features, n_test = 1500, 40, 400
    X, y = sk_make(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=15,
        n_classes=4,
        random_state=3,
    )
    X = X.astype(np.float32)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    print(f"\n=== ternary PTQ ablation ({backend}, {n_train}x{n_features}) ===")
    records: list[dict[str, Any]] = []
    for label, quant, granularity in QUANT_VARIANTS:
        common: dict[str, Any] = {
            "backend": backend,
            "n_estimators": args.n_estimators,
            "random_state": 42,
            "batch_size": args.batch_size,
            "quantization": quant,
            "quantization_granularity": granularity or "per_row",
        }
        model = TabICLClassifier(**common)
        extra: dict[str, Any] = {"label": label}
        if quant == "ternary":
            try:
                extra["ptq_sparsity"] = round(
                    _ptq_sparsity(CLASSIFIER_CHECKPOINT, granularity), 4
                )
            except Exception:  # noqa: BLE001 - sparsity is informational
                pass
        records.append(
            _fit_predict_record(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                task="classification",
                backend=backend,
                repeat=args.repeat,
                regression=False,
                quiet_warnings=True,
                extra=extra,
            )
        )
    return records


def bench_cache_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Predict time across batch_size x kv_cache combos on a fixed case."""
    backend = args.backend
    n_train, n_features, n_test = 1000, 20, 200
    X, y = make_classification(n_train + n_test, n_features)
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    print(f"\n=== batch-size / KV-cache sweep ({backend}, {n_train}x{n_features}) ===")
    records: list[dict[str, Any]] = []
    for kv_cache in (False, True):
        for batch_size in (8, 32, 128):
            model = TabICLClassifier(
                backend=backend,
                n_estimators=args.n_estimators,
                random_state=42,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            label = f"bs={batch_size:<3d} kv_cache={str(kv_cache).lower()}"
            records.append(
                _fit_predict_record(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    task="classification",
                    backend=backend,
                    repeat=args.repeat,
                    regression=False,
                    extra={"label": label},
                )
            )
    return records


def bench_upstream(task: str, n_samples: int, n_features: int, args) -> None:
    try:
        from tabicl import TabICLClassifier as UpstreamClassifier
        from tabicl import TabICLRegressor as UpstreamRegressor
    except ImportError:
        print("  upstream tabicl not installed; skipping")
        return
    n_test = min(1000, max(200, n_samples // 10))
    if task == "regression":
        X, y = make_regression(n_samples + n_test, n_features)
        model = UpstreamRegressor()
    elif task == "classification":
        X, y = make_classification(n_samples + n_test, n_features)
        model = UpstreamClassifier()
    else:
        X, y = make_mixed_classification(n_samples + n_test, n_features)
        model = UpstreamClassifier()
    X_train, y_train, X_test = X[:-n_test], y[:-n_test], X[-n_test:]
    try:
        fit_time, fit_std = _timed(lambda: model.fit(X_train, y_train), args.repeat)

        def predict():
            return model.predict(X_test)

        predict_time, predict_std = _timed(predict, args.repeat)
        score = model.score(X_test, y[-n_test:])
    except Exception as exc:  # noqa: BLE001
        print(f"  {task:>18} upstream FAILED: {exc}")
        return
    print(
        f"  {task:>18} {n_samples:>6}x{n_features:<4} "
        f"fit {fit_time:7.3f}s ±{fit_std:.3f}  "
        f"predict {predict_time:7.3f}s ±{predict_std:.3f}  "
        f"score {score:.3f}  [upstream]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small grid")
    parser.add_argument("--smoke", action="store_true", help="write suffixed JSON")
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats")
    parser.add_argument("--n-estimators", type=int, default=8, help="ensemble members")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--quant-ablation", action="store_true")
    parser.add_argument("--cache-sweep", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch", "mojo"),
        default="numpy",
        help="compute backend to benchmark",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--with-upstream",
        action="store_true",
        help="also compare against pip tabicl (needs tabicl-bench extra)",
    )
    args = parser.parse_args()

    json_out = args.json_out
    if args.smoke:
        json_out = json_out.with_suffix(".smoke.json")

    if args.quick:
        sizes = [(300, 10)]
    else:
        sizes = [(300, 10), (1000, 10), (1000, 100), (5000, 100)]
    tasks = ["classification", "regression", "mixed categorical"]

    print(
        f"TabICL benchmark (backend={args.backend}, "
        f"n_estimators={args.n_estimators}, kv_cache={args.kv_cache})"
    )
    records: list[dict[str, Any]] = []
    for n_samples, n_features in sizes:
        print(f"\n--- dataset {n_samples} x {n_features} ---")
        for task in tasks:
            records.append(bench_case(task, n_samples, n_features, args))
            if args.with_upstream:
                bench_upstream(task, n_samples, n_features, args)

    if args.quant_ablation:
        records.extend(bench_quant_ablation(args))
    if args.cache_sweep:
        records.extend(bench_cache_sweep(args))

    save_results(json_out, args.backend, _env_meta(args), records)
    print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
