#!/usr/bin/env python3
"""Benchmark TabM training: NumPy reference vs Mojo kernels.

Times ``TabMRegressor`` / ``TabMClassifier`` fits on synthetic datasets
(continuous-only and mixed continuous/categorical) for both backends.
A PyTorch reference (upstream TabM) can be added with ``--with-torch``
when the optional benchmark dependencies are installed:

    uv sync --group tabm-bench

Usage:
    python scripts/benchmarks/bench_tabm.py [--max-iter N] [--with-torch]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from shinrin.tabm import TabMClassifier, TabMRegressor


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
    y = (logits + 0.5 * rng.randn(n_samples)).argsort().argsort() % 3
    return X, y.astype(np.int64)


def make_mixed_classification(n_samples: int, n_features: int, seed: int = 2):
    """Continuous + low-cardinality 'categorical' columns (openml-ish)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Round half the columns to a few levels so they are detected as
    # categorical by the preprocessing pipeline.
    n_cat = n_features // 2
    for j in range(n_cat):
        X[:, j] = np.round(X[:, j] * 2.0) / 2.0  # ~6 levels
    logits = (
        X[:, :n_cat].sum(axis=1)
        + 0.5 * X[:, n_cat:].sum(axis=1)
        + 0.3 * rng.randn(n_samples)
    )
    y = (logits > np.median(logits)).astype(np.int64)
    return X, y


def bench_backend(backend: str, max_iter: int, n_samples: int, n_features: int):
    print(f"\n--- backend: {backend} ---")
    cases = [
        ("regression", make_regression(n_samples, n_features)),
        ("binary+multiclass", make_classification(n_samples, n_features)),
        ("mixed categorical", make_mixed_classification(n_samples, n_features)),
    ]
    for name, (X, y) in cases:
        if name == "regression":
            model = TabMRegressor(max_iter=max_iter, random_state=0, solver="adam")
        else:
            model = TabMClassifier(max_iter=max_iter, random_state=0, solver="adam")
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
        score = model.score(X, y)
        print(
            f"  {name:20s} fit {elapsed:7.2f}s  "
            f"score {score:.4f}  ({n_samples}x{X.shape[1]}, {max_iter} epochs)"
        )


def _remap_categorical(X: np.ndarray, n_cat: int):
    """Encode each categorical column as contiguous integer codes."""
    codes = np.empty((X.shape[0], n_cat), dtype=np.int64)
    cards = []
    for j in range(n_cat):
        uniq, inv = np.unique(X[:, j], return_inverse=True)
        codes[:, j] = inv
        cards.append(int(uniq.size))
    return codes, cards


def _prepare_numeric(X_num: np.ndarray, n_bins: int = 64, d_embedding: int = 8):
    """Mirror the shinrin numeric pipeline: asinh -> standardize -> PLE bins.

    Returns ``(x_tensor, piecewise_linear_embeddings_or_None)`` using the
    same ``n_bins`` / ``d_embedding`` defaults as ``TabMRegressor``.
    """
    import torch
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings

    xn = np.arcsinh(X_num.astype(np.float64))
    std = xn.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    xn = ((xn - xn.mean(axis=0)) / std).astype(np.float32)
    probs = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for j in range(xn.shape[1]):
        edges = np.unique(np.quantile(xn[:, j], probs))
        if len(edges) < 2:
            center = float(edges[0])
            edges = np.array([center - 0.1, center + 0.1])
        elif len(edges) < n_bins + 1:
            edges = np.linspace(edges[0], edges[-1], n_bins + 1)
        bins.append(torch.tensor(edges, dtype=torch.float32))
    return torch.tensor(xn), PiecewiseLinearEmbeddings(
        bins, d_embedding=d_embedding, activation=False, version="B"
    )


def bench_torch(max_iter: int, n_samples: int, n_features: int):
    """Reference timings with the upstream PyTorch TabM (optional).

    Mirrors the NumPy/Mojo setup for a like-for-like comparison:
    matched backbone (``d_block=256``, ``n_blocks=1``, ``k=32``,
    ``arch_type='tabm'``, dropout 0.1), Adam at the same learning rate,
    minibatches of 200, and proper per-member task losses (mean of the
    k member losses, not the loss of the mean prediction). Numeric
    features get the same treatment as in shinrin: asinh +
    standardization followed by piecewise-linear embeddings with 64
    quantile bins (``d_embedding=8``). The timed region covers the
    training loop only; preprocessing and scoring are excluded from the
    timing but scores are reported as a sanity check.
    """
    try:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import r2_score

        from tabm import TabM
    except ImportError as exc:
        print(f"\n--- torch reference unavailable ({exc}) ---")
        return
    print("\n--- backend: torch (upstream reference) ---")
    cases = [
        ("regression", make_regression(n_samples, n_features)),
        ("binary+multiclass", make_classification(n_samples, n_features)),
        ("mixed categorical", make_mixed_classification(n_samples, n_features)),
    ]
    for name, (X, y) in cases:
        try:
            torch.manual_seed(0)
            n_cat = X.shape[1] // 2 if name == "mixed categorical" else 0
            x_cat_codes, cat_cards = (
                _remap_categorical(X, n_cat) if n_cat else (None, [])
            )
            n_num = X.shape[1] - n_cat
            d_out = 1 if name == "regression" else int(y.max()) + 1
            xt_num, num_embeddings = _prepare_numeric(X[:, n_cat:])
            model = TabM.make(
                n_num_features=n_num,
                cat_cardinalities=cat_cards or None,
                d_out=d_out,
                num_embeddings=num_embeddings,
                # Match the shinrin default architecture
                # (hidden_layer_sizes=(256,) -> one block, width 256).
                d_block=256,
                n_blocks=1,
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            xt_cat = torch.tensor(x_cat_codes, dtype=torch.long) if n_cat else None
            yt = torch.tensor(
                y, dtype=torch.float32 if name == "regression" else torch.long
            )
            n = len(yt)
            bs = min(200, n)

            def forward(xn, xc):
                return model(xn) if xc is None else model(xn, xc)

            t0 = time.perf_counter()
            model.train()
            for _ in range(max_iter):
                perm = torch.randperm(n)
                for start in range(0, n, bs):
                    idx = perm[start : start + bs]
                    opt.zero_grad()
                    out = forward(
                        xt_num[idx], xt_cat[idx] if xt_cat is not None else None
                    )
                    if name == "regression":
                        # mean over batch x k == mean of per-member MSEs
                        loss = F.mse_loss(
                            out.squeeze(-1),
                            yt[idx][:, None].expand(-1, out.shape[1]),
                        )
                    else:
                        loss = F.cross_entropy(
                            out.flatten(0, 1),
                            yt[idx].repeat_interleave(out.shape[1]),
                        )
                    loss.backward()
                    opt.step()
            elapsed = time.perf_counter() - t0

            model.eval()
            with torch.no_grad():
                out = forward(xt_num, xt_cat)
                if name == "regression":
                    score = r2_score(yt.numpy(), out.mean(dim=1).squeeze(-1).numpy())
                else:
                    prob = F.softmax(out, dim=-1).mean(dim=1)
                    score = float((prob.argmax(dim=-1) == yt).float().mean())
            print(
                f"  {name:20s} fit {elapsed:7.2f}s  "
                f"score {score:.4f}  ({n_samples}x{X.shape[1]}, {max_iter} epochs)"
            )
        except Exception as exc:
            print(f"  {name:20s} skipped ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument(
        "--backends", default="numpy,mojo", help="comma list: numpy,mojo"
    )
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="also time the upstream PyTorch reference (needs tabm-bench deps)",
    )
    args = parser.parse_args()

    print(
        f"TabM benchmark: {args.samples} samples x {args.features} features, "
        f"{args.max_iter} Adam epochs"
    )
    for backend in args.backends.split(","):
        backend = backend.strip()
        if not backend:
            continue
        try:
            os.environ["SHINRIN_TABM_BACKEND"] = backend
            bench_backend(backend, args.max_iter, args.samples, args.features)
        except Exception as exc:
            print(f"\n--- backend: {backend} unavailable ({exc}) ---")
    if args.with_torch:
        bench_torch(args.max_iter, args.samples, args.features)


if __name__ == "__main__":
    main()
