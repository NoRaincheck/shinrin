"""Tests for the shinrin.benchmark utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from shinrin.benchmark import (
    ablation_benchmark,
    benchmark_training,
    full_benchmark,
    print_ablation_report,
    print_benchmark_report,
)
from shinrin.mlp import MLPClassifier


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
        "fp": MLPClassifier(hidden_layer_sizes=(8,), max_iter=30, random_state=0),
        "ternary": MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=30,
            random_state=0,
            quantization="ternary",
        ),
    }
    results = ablation_benchmark(
        variants, X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]
    )
    assert set(results) == {"fp", "ternary"}
    for metrics in results.values():
        assert metrics["fit_time"] > 0
        assert metrics["predict_time"] >= 0
        assert 0 <= metrics["train_score"] <= 1
        assert 0 <= metrics["test_score"] <= 1
    # the quantized variant must not be catastrophically worse on this
    # linearly separable toy (ablation-style sanity bound)
    assert results["ternary"]["test_score"] >= results["fp"]["test_score"] - 0.15


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
