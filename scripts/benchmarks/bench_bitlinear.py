#!/usr/bin/env python3
"""Benchmark ternary (BitLinear) quantization against full precision.

Times fits of the shinrin MLP and TabM estimators in three variants

- ``fp``            : full-precision latent weights (baseline)
- ``ternary/row``   : training-aware ternary quantization, per-row scales
- ``ternary/tensor``: training-aware ternary quantization, per-tensor scale

and reports fit time, train score and the effective weight sparsity
(fraction of exactly-zero effective weights induced by the ternary
scheme). The MLP additionally runs ``ternary+out``, which also quantizes
the output layer. Each estimator runs on the NumPy backend and, when
built, the Mojo kernels. With ``--tabicl`` the experimental TabICL
post-training quantization is evaluated for inference time and held-out
accuracy.

Usage:
    python scripts/benchmarks/bench_bitlinear.py [--samples N]
        [--features N] [--max-iter N] [--backends numpy,mojo]
        [--estimators mlp,tabm] [--tasks cls,reg] [--tabicl]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def make_classification(n_samples: int, n_features: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    logits = X @ rng.randn(n_features).astype(np.float32)
    y = (logits + 0.5 * rng.randn(n_samples)).argsort().argsort() % 3
    return X, y.astype(np.int64)


def make_regression(n_samples: int, n_features: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features).astype(np.float32)
    y = X @ w * 20.0 + 2.0 * rng.randn(n_samples).astype(np.float32)
    return X, ((y - y.mean()) / y.std()).astype(np.float32)


def _mlp_sparsity(clf) -> float:
    """Fraction of exactly-zero effective weights over quantized layers."""
    from shinrin._quant import ternary_quantize_dequantize

    cfg = clf.config_
    if cfg.quantization == "none":
        return 0.0
    arrays = clf.params_.arrays
    total = zeros = 0
    for i in range(cfg.n_layers):
        if not cfg.layer_is_quantized(i):
            continue
        w = arrays[f"l{i}_w"]
        total += w.size
        q = ternary_quantize_dequantize(w, cfg.quantization_granularity)
        zeros += int((q == 0).sum())
    return zeros / max(total, 1)


def _tabm_sparsity(clf) -> float:
    """Zero fraction over quantized shared-block weights (head excluded)."""
    from shinrin._quant import ternary_quantize_dequantize

    arrays = clf.params_.arrays
    blk = {k: v for k, v in arrays.items() if k.startswith("blk") and k.endswith("_w")}
    if not blk or clf.config_.quantization == "none":
        return 0.0
    total = sum(int(v.size) for v in blk.values())
    zeros = sum(
        int(
            (
                ternary_quantize_dequantize(v, clf.config_.quantization_granularity)
                == 0
            ).sum()
        )
        for v in blk.values()
    )
    return zeros / max(total, 1)


def _mlp_variants():
    return [
        ("fp", {}),
        (
            "ternary/row",
            {"quantization": "ternary", "quantization_granularity": "per_row"},
        ),
        (
            "ternary/tensor",
            {"quantization": "ternary", "quantization_granularity": "per_tensor"},
        ),
        (
            "ternary+out",
            {
                "quantization": "ternary",
                "quantization_granularity": "per_row",
                "quantize_output": True,
            },
        ),
    ]


def _tabm_variants():
    return [
        ("fp", {}),
        (
            "ternary/row",
            {"quantization": "ternary", "quantization_granularity": "per_row"},
        ),
        (
            "ternary/tensor",
            {"quantization": "ternary", "quantization_granularity": "per_tensor"},
        ),
    ]


def bench_mlp(X, y, max_iter: int, task: str) -> None:
    from shinrin.mlp import MLPClassifier, MLPRegressor

    cls = MLPClassifier if task == "cls" else MLPRegressor
    for name, kw in _mlp_variants():
        model = cls(
            hidden_layer_sizes=(128, 64), max_iter=max_iter, random_state=0, **kw
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
        extra = f"  eff-zero {_mlp_sparsity(model):6.1%}" if kw else ""
        print(f"  {name:15s} fit {elapsed:7.2f}s  score {model.score(X, y):.4f}{extra}")


def bench_tabm(X, y, max_iter: int, task: str) -> None:
    from shinrin.tabm import TabMClassifier, TabMRegressor

    cls = TabMClassifier if task == "cls" else TabMRegressor
    for name, kw in _tabm_variants():
        model = cls(
            hidden_layer_sizes=(128, 64),
            k=8,
            max_iter=max_iter,
            random_state=0,
            solver="adam",
            use_embeddings=False,
            alpha=0.0,
            **kw,
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
        extra = f"  eff-zero {_tabm_sparsity(model):6.1%}" if kw else ""
        print(f"  {name:15s} fit {elapsed:7.2f}s  score {model.score(X, y):.4f}{extra}")


def bench_tabicl(n_samples: int, n_test: int, seed: int = 3) -> None:
    """PTQ inference benchmark on the cached classifier checkpoint."""
    import warnings

    from sklearn.datasets import make_classification as sk_make
    from sklearn.model_selection import train_test_split

    from shinrin import TabICLClassifier

    X, y = sk_make(
        n_samples=n_samples + n_test,
        n_features=30,
        n_informative=12,
        random_state=seed,
    )
    X = X.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=n_test, random_state=seed)

    print("\n=== tabicl classifier (post-training, inference) ===")
    for quant in ("none", "ternary"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TabICLClassifier(
                backend="numpy",
                random_state=0,
                allow_auto_download=False,
                quantization=quant,
            ).fit(Xtr, ytr)
            t0 = time.perf_counter()
            pred = clf.predict(Xte)
            elapsed = time.perf_counter() - t0
        acc = float((pred == yte).mean())
        label = "fp" if quant == "none" else "ternary/row"
        print(f"  {label:15s} predict {elapsed:6.2f}s  held-out acc {acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--backends", default="numpy,mojo")
    parser.add_argument("--estimators", default="mlp,tabm")
    parser.add_argument("--tasks", default="cls,reg")
    parser.add_argument("--tabicl", action="store_true")
    parser.add_argument("--tabicl-samples", type=int, default=600)
    parser.add_argument("--tabicl-test", type=int, default=250)
    args = parser.parse_args()

    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    datasets = {
        "cls": make_classification(args.samples, args.features),
        "reg": make_regression(args.samples, args.features),
    }

    print(
        f"BitLinear benchmark: {args.samples} samples x {args.features} features, "
        f"{args.max_iter} Adam epochs"
    )

    try:
        for backend in backends:
            os.environ["SHINRIN_MLP_BACKEND"] = backend
            os.environ["SHINRIN_TABM_BACKEND"] = backend
            print(f"\n--- backend: {backend} ---")
            for est in estimators:
                for task in tasks:
                    X, y = datasets[task]
                    label = f"=== {est} {task} ==="
                    print(label)
                    fn = bench_mlp if est == "mlp" else bench_tabm
                    try:
                        fn(X, y, args.max_iter, task)
                    except Exception as exc:
                        print(f"  skipped ({type(exc).__name__}: {exc})")
        if args.tabicl:
            try:
                bench_tabicl(args.tabicl_samples, args.tabicl_test)
            except Exception as exc:
                print(f"tabicl section skipped ({type(exc).__name__}: {exc})")
    finally:
        os.environ.pop("SHINRIN_MLP_BACKEND", None)
        os.environ.pop("SHINRIN_TABM_BACKEND", None)


if __name__ == "__main__":
    main()
