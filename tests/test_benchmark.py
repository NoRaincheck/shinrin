"""Tests for the shinrin.benchmark utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from shinrin.benchmark import (
    ablation_benchmark,
    benchmark_training,
    full_benchmark,
    print_ablation_report,
    print_benchmark_report,
)


def _toy_data(n=150, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    return X, y


def test_benchmark_training_and_full():
    X, y = _toy_data()
    models = {"lr": LogisticRegression(max_iter=100)}
    train = benchmark_training(models, X, y, n_repeats=2)
    assert set(train) == {"lr"}
    for key in ("mean_time", "min_time", "max_time", "std_time"):
        assert key in train["lr"]

    results = full_benchmark(models, X, y, X, n_repeats_train=1)
    assert "lr.train" in results and "lr.predict" in results and "lr.size" in results
    print_benchmark_report(results)


def test_ablation_benchmark_metrics():
    X, y = _toy_data()
    n_test = 40
    variants = {
        "ridge-c": RidgeClassifier(alpha=1.0),
        "ridge-strong": RidgeClassifier(alpha=100.0),
    }
    results = ablation_benchmark(
        variants, X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]
    )
    assert set(results) == {"ridge-c", "ridge-strong"}
    for metrics in results.values():
        assert metrics["fit_time"] > 0
        assert metrics["predict_time"] >= 0
        assert 0 <= metrics["train_score"] <= 1
        assert 0 <= metrics["test_score"] <= 1
    # both variants must be reasonable on this linearly separable toy
    assert results["ridge-c"]["test_score"] >= 0.7
    assert results["ridge-strong"]["test_score"] >= 0.7


def test_print_ablation_report(capsys):
    results = {
        "fp": {
            "fit_time": 0.10,
            "predict_time": 1e-5,
            "train_score": 0.95,
            "test_score": 0.90,
        },
        "ternary/row": {
            "fit_time": 0.12,
            "predict_time": 1e-5,
            "train_score": 0.90,
            "test_score": 0.85,
        },
    }
    print_ablation_report(results)
    out = capsys.readouterr().out
    assert "variant" in out
    assert "fp" in out and "ternary/row" in out
    assert "+0.0000" in out  # baseline delta row
    assert "-0.0500" in out  # ternary delta vs fp
    print_ablation_report({})  # empty input is a no-op
