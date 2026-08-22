#!/usr/bin/env python3
"""Benchmark TabICL inference: NumPy vs torch backends.

Times ``TabICLClassifier`` / ``TabICLRegressor`` fit (preprocessing +
optional KV-cache build) and predict (ensemble forward passes) separately
on synthetic datasets. The upstream ``tabicl`` package can be compared
with ``--with-upstream`` when the benchmark extra is installed:

    uv sync --extra tabicl-bench

With ``--mojo`` the raw end-to-end forward pass of the native Mojo kernels
is timed against the torch backend (CPU and MPS where available) on the
real classifier checkpoint weights. The Mojo kernels are still a reduced
scaffold — they read only the first feature group of the training rows,
skip test-row encoding, and aggregate ICL into class embeddings instead of
attending over train rows — so those numbers are throughput indicators for
the current kernels, not like-for-like backend comparisons (see
``scripts/benchmarks/TABICL_BENCHMARK.md``).

Usage:
    python scripts/benchmarks/bench_tabicl.py [--quick] [--repeat N]
        [--with-upstream] [--kv-cache] [--mojo]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from shinrin.tabicl import TabICLClassifier, TabICLRegressor


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


def bench_case(task: str, n_samples: int, n_features: int, args) -> None:
    backend = "numpy" if not args.torch else "torch"
    n_test = min(1000, max(200, n_samples // 10))
    if task == "regression":
        X, y = make_regression(n_samples + n_test, n_features)
    elif task == "classification":
        X, y = make_classification(n_samples + n_test, n_features)
    else:
        X, y = make_mixed_classification(n_samples + n_test, n_features)
    X_train, y_train, X_test = X[:-n_test], y[:-n_test], X[-n_test:]

    common: dict[str, Any] = {
        "backend": backend,
        "n_estimators": args.n_estimators,
        "random_state": 42,
        "kv_cache": args.kv_cache,
        "batch_size": args.batch_size,
    }
    model = (
        TabICLRegressor(**common) if task == "regression" else TabICLClassifier(**common)
    )

    fit_time = predict_time = fit_std = predict_std = float("nan")
    score = float("nan")
    try:
        fit_time, fit_std = _timed(lambda: model.fit(X_train, y_train), args.repeat)

        def predict():
            if task == "regression":
                return model.predict(X_test)
            return model.predict_proba(X_test)

        predict_time, predict_std = _timed(predict, args.repeat)
        score = model.score(X_test, y[-n_test:])
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        print(f"  {task:>18} FAILED: {exc}")
        return

    print(
        f"  {task:>18} {n_samples:>6}x{n_features:<4} "
        f"fit {fit_time:7.3f}s ±{fit_std:.3f}  "
        f"predict {predict_time:7.3f}s ±{predict_std:.3f}  "
        f"score {score:.3f}  [{backend}]"
    )


KERNEL_SIZES = ((300, 150), (500, 200), (2000, 200), (5000, 500))


def _kernel_inputs(
    n_train: int, n_test: int, n_features: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic input shared by the parent and child processes."""
    rng = np.random.RandomState(7)
    X = rng.randn(n_train + n_test, n_features).astype(np.float32)
    y_train = rng.randint(0, max(int(n_classes), 2), size=n_train)
    return X, y_train


def _mojo_child(
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    out_q,
) -> None:
    """Time ONE raw Mojo forward pass and exit.

    Each timed sample therefore runs in a pristine process: the kernels
    still have a known memory-corruption issue whose manifestation depends
    on heap layout (see TABICL_BENCHMARK.md), so isolating every forward
    keeps repeated sampling meaningful.
    """
    import time as _time

    from shinrin._tabicl._checkpoint import CLASSIFIER_V2, ensure_npz
    from shinrin._tabicl._config import TabICLConfig
    from shinrin._tabicl._mojo_backend import TabICLMojoModel

    X, y_train = _kernel_inputs(n_train, n_test, n_features, n_classes)
    _, config_dict, params = ensure_npz(filename=CLASSIFIER_V2)
    model = TabICLMojoModel(TabICLConfig.from_dict(config_dict), params)

    t0 = _time.perf_counter()
    model.forward(X, y_train)
    out_q.put(_time.perf_counter() - t0)


def bench_mojo(args) -> None:
    """Raw end-to-end forward timings: Mojo kernels vs torch (CPU/MPS).

    The Mojo backend only exposes a single end-to-end ``forward`` and the
    kernels are a reduced scaffold (see TABICL_BENCHMARK.md), so each Mojo
    cell is timed inside a throwaway subprocess: the sweep survives the
    memory corruption the kernels still exhibit at larger inputs.
    """
    import multiprocessing as mp

    import torch

    from shinrin._tabicl._checkpoint import CLASSIFIER_V2, ensure_npz
    from shinrin._tabicl._config import TabICLConfig
    from shinrin._tabicl._model_torch import TabICLTorchModel

    _, config_dict, params = ensure_npz(filename=CLASSIFIER_V2)
    config = TabICLConfig.from_dict(config_dict)
    n_classes = max(int(config.max_classes), 2)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    torch_models = {
        dev: TabICLTorchModel(config, params, device=dev) for dev in devices
    }

    def time_cell(label: str, fn) -> str:
        try:
            mean, std = _timed(fn, args.repeat)
        except Exception as exc:  # noqa: BLE001 - report and continue
            return f"{label} FAILED ({exc})"
        return f"{label} {mean:.3f}s ±{std:.3f}"

    ctx = mp.get_context("spawn")
    print(
        f"\nTabICL kernel benchmark (raw forward, "
        f"n_features={args.n_features}, repeat={args.repeat})"
    )
    for n_train, n_test in KERNEL_SIZES:
        cells = []

        # Mojo: one sample per child process; retry until enough samples
        # survive (crashes are expected and contained) or attempts run out.
        samples: list[float] = []
        for _ in range(args.repeat * 8):
            if len(samples) >= args.repeat:
                break
            q = ctx.Queue()
            proc = ctx.Process(
                target=_mojo_child,
                args=(n_train, n_test, args.n_features, n_classes, q),
            )
            proc.start()
            proc.join(timeout=600)
            if proc.exitcode == 0 and not q.empty():
                samples.append(float(q.get()))
            else:
                proc.terminate()
        if samples:
            arr = np.asarray(samples)
            cells.append(
                f"mojo {arr.mean():.3f}s ±{arr.std():.3f} "
                f"(n={len(samples)}/{args.repeat * 8})"
            )
        else:
            cells.append("mojo CRASHED")

        X, y_train = _kernel_inputs(n_train, n_test, args.n_features, n_classes)
        for dev, model in torch_models.items():
            cells.append(
                time_cell(f"torch-{dev}", lambda m=model: m.forward(X, y_train))
            )
        print(f"  train {n_train:>5} x test {n_test:<4} " + " | ".join(cells))


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
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats")
    parser.add_argument(
        "--n-estimators", type=int, default=8, help="ensemble members"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument(
        "--torch", action="store_true", help="benchmark the torch backend"
    )
    parser.add_argument(
        "--mojo",
        action="store_true",
        help="raw forward-pass timings: Mojo kernels vs torch (CPU/MPS)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=100,
        help="features per row for --mojo kernel timings",
    )
    parser.add_argument(
        "--with-upstream",
        action="store_true",
        help="also compare against pip tabicl (needs tabicl-bench extra)",
    )
    args = parser.parse_args()

    if args.mojo:
        bench_mojo(args)
        return

    if args.quick:
        sizes = [(300, 10)]
    else:
        sizes = [(300, 10), (1000, 10), (1000, 100), (5000, 100)]
    tasks = ["classification", "regression", "mixed categorical"]

    print(f"TabICL benchmark (backend={'torch' if args.torch else 'numpy'}, "
          f"n_estimators={args.n_estimators}, kv_cache={args.kv_cache})")
    for n_samples, n_features in sizes:
        print(f"\n--- dataset {n_samples} x {n_features} ---")
        for task in tasks:
            bench_case(task, n_samples, n_features, args)
            if args.with_upstream:
                bench_upstream(task, n_samples, n_features, args)


if __name__ == "__main__":
    main()
