"""Benchmarking utilities for shinrin tree and forest models.

This module provides functions to benchmark training speed, prediction speed,
and model size of shinrin models against scikit-learn equivalents.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def benchmark_training(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 3,
) -> dict[str, dict[str, Any]]:
    """Benchmark training speed of multiple models.

    Parameters
    ----------
    models : dict of str → estimator instance
        Mapping of model names to fitted or unfitted estimators.
        Unfitted models will be fitted during benchmarking.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    n_repeats : int
        Number of times to repeat training for averaging.

    Returns
    -------
    dict
        Mapping of model names to timing results with keys:
        - ``mean_time``: mean training time in seconds
        - ``min_time``: minimum training time in seconds
        - ``max_time``: maximum training time in seconds
        - ``std_time``: standard deviation of training times

    Examples
    --------
    >>> from shinrin import MondrianTreeRegressor
    >>> from shinrin.benchmark import benchmark_training
    >>> import numpy as np
    >>> X = np.random.randn(1000, 10).astype(np.float32)
    >>> y = np.random.randn(1000)
    >>> models = {"shinrin": MondrianTreeRegressor(random_state=0)}
    >>> results = benchmark_training(models, X, y)
    >>> print(results["shinrin"]["mean_time"])  # doctest: +SKIP
    0.0123
    """
    results = {}

    for name, model in models.items():
        times = []
        for _ in range(n_repeats):
            # Clone the model to ensure fresh state
            try:
                from sklearn.base import clone

                clone_model = clone(model)
            except RuntimeError:
                clone_model = model

            start = time.perf_counter()
            clone_model.fit(X, y)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        results[name] = {
            "mean_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
        }

    return results


def benchmark_prediction(
    models: dict[str, Any],
    X: np.ndarray,
    n_repeats: int = 100,
) -> dict[str, dict[str, Any]]:
    """Benchmark prediction speed of fitted models.

    Parameters
    ----------
    models : dict of str → fitted estimator instance
        Mapping of model names to fitted estimators.
    X : ndarray of shape (n_samples, n_features)
        Test data for prediction.
    n_repeats : int
        Number of prediction runs to average over.

    Returns
    -------
    dict
        Mapping of model names to timing results with keys:
        - ``mean_time``: mean prediction time in seconds
        - ``min_time``: minimum prediction time in seconds
        - ``max_time``: maximum prediction time in seconds
        - ``std_time``: standard deviation of prediction times
        - ``predictions_shape``: shape of the prediction output

    Examples
    --------
    >>> from shinrin import MondrianTreeRegressor
    >>> from shinrin.benchmark import benchmark_prediction
    >>> import numpy as np
    >>> X = np.random.randn(100, 5).astype(np.float32)
    >>> y = np.random.randn(100)
    >>> tree = MondrianTreeRegressor(random_state=0)
    >>> tree.fit(X, y)
    >>> models = {"shinrin": tree}
    >>> results = benchmark_prediction(models, X)
    >>> print(results["shinrin"]["mean_time"])  # doctest: +SKIP
    0.0001
    """
    results = {}

    for name, model in models.items():
        times = []
        predictions = None

        for _ in range(n_repeats):
            start = time.perf_counter()
            predictions = model.predict(X)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        results[name] = {
            "mean_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "predictions_shape": list(predictions.shape)
            if predictions is not None
            else [],
        }

    return results


def benchmark_model_size(
    models: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Compare model sizes (number of parameters / nodes).

    Parameters
    ----------
    models : dict of str → fitted estimator instance
        Mapping of model names to fitted estimators.

    Returns
    -------
    dict
        Mapping of model names to size metrics with keys:
        - ``n_nodes``: total number of tree nodes
        - ``n_leaves``: total number of leaf nodes
        - ``n_estimators``: number of trees in the model (1 for single trees)

    Examples
    --------
    >>> from shinrin import MondrianTreeRegressor, RandomForestRegressor
    >>> from shinrin.benchmark import benchmark_model_size
    >>> import numpy as np
    >>> X = np.random.randn(100, 5).astype(np.float32)
    >>> y = np.random.randn(100)
    >>> tree = MondrianTreeRegressor(random_state=0).fit(X, y)
    >>> rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    >>> models = {"tree": tree, "forest": rf}
    >>> results = benchmark_model_size(models)
    >>> print(results["tree"]["n_estimators"])  # doctest: +SKIP
    1
    """
    results = {}

    for name, model in models.items():
        if hasattr(model, "tree_"):
            # Single tree
            t = model.tree_
            n_nodes = int(t.node_count)
            n_leaves = int(np.sum(t.children_left == -1))
            results[name] = {
                "n_nodes": n_nodes,
                "n_leaves": n_leaves,
                "n_estimators": 1,
            }
        elif hasattr(model, "estimators_"):
            # Forest
            total_nodes = 0
            total_leaves = 0
            for tree in model.estimators_:
                if hasattr(tree, "tree_"):
                    t = tree.tree_
                    total_nodes += int(t.node_count)
                    total_leaves += int(np.sum(t.children_left == -1))

            results[name] = {
                "n_nodes": total_nodes,
                "n_leaves": total_leaves,
                "n_estimators": len(model.estimators_),
            }
        else:
            results[name] = {
                "n_nodes": 0,
                "n_leaves": 0,
                "n_estimators": 0,
            }

    return results


def full_benchmark(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_repeats_train: int = 3,
    n_repeats_predict: int = 100,
) -> dict[str, dict[str, Any]]:
    """Run a complete benchmark suite on multiple models.

    Parameters
    ----------
    models : dict of str → estimator instance
        Mapping of model names to estimators. Unfitted models will be
        fitted during benchmarking.
    X_train : ndarray of shape (n_samples, n_features)
        Training data.
    y_train : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    X_test : ndarray of shape (n_samples, n_features)
        Test data for prediction benchmarking.
    n_repeats_train : int
        Number of training repeats.
    n_repeats_predict : int
        Number of prediction repeats.

    Returns
    -------
    dict
        Nested dictionary with benchmark results:
        - ``{name}.train``: training timing results
        - ``{name}.predict``: prediction timing results
        - ``{name}.size``: model size metrics

    Examples
    --------
    >>> from shinrin import MondrianTreeRegressor
    >>> from shinrin.benchmark import full_benchmark
    >>> import numpy as np
    >>> X = np.random.randn(500, 5).astype(np.float32)
    >>> y = np.random.randn(500)
    >>> X_test = X[:50]
    >>> models = {"shinrin": MondrianTreeRegressor(random_state=0)}
    >>> results = full_benchmark(models, X, y, X_test)
    >>> "shinrin" in results  # doctest: +SKIP
    True
    """
    all_results = {}

    # Training benchmark
    train_results = benchmark_training(
        models,
        X_train,
        y_train,
        n_repeats=n_repeats_train,
    )
    for name, metrics in train_results.items():
        all_results[f"{name}.train"] = metrics

    # Fit models for prediction benchmark
    for model in models.values():
        if not hasattr(model, "tree_"):
            model.fit(X_train, y_train)

    # Prediction benchmark (requires fitted models)
    predict_results = benchmark_prediction(
        models,
        X_test,
        n_repeats=n_repeats_predict,
    )
    for name, metrics in predict_results.items():
        all_results[f"{name}.predict"] = metrics

    # Model size
    size_results = benchmark_model_size(models)
    for name, metrics in size_results.items():
        all_results[f"{name}.size"] = metrics

    return all_results


def ablation_benchmark(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats_predict: int = 100,
) -> dict[str, dict[str, Any]]:
    """Measure fit time and held-out quality for a set of model variants.

    The intended use is *ablation* benchmarks: pass the same estimator
    with one feature toggled — e.g. a forest with and without a preprocessing
    step — to quantify the before/after cost of the feature on speed and
    quality.

    Parameters
    ----------
    models : dict of str → estimator instance
        Mapping of variant names to unfitted estimators. Every entry is
        fitted exactly once on ``(X_train, y_train)``; use distinct
        instances per variant.
    X_train : ndarray of shape (n_samples, n_features)
        Training data.
    y_train : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Training targets.
    X_test : ndarray of shape (n_test_samples, n_features)
        Held-out data used for the quality measurement.
    y_test : ndarray of shape (n_test_samples,) or (n_test_samples, n_outputs)
        Held-out targets.
    n_repeats_predict : int
        Number of prediction repeats for the timing average.

    Returns
    -------
    dict
        Mapping of variant names to metrics with keys:
        - ``fit_time``: wall-clock ``fit()`` seconds
        - ``predict_time``: mean prediction seconds over ``n_repeats_predict``
        - ``train_score``: estimator ``score`` on the training data
        - ``test_score``: estimator ``score`` on the held-out data

    Examples
    --------
    >>> from shinrin import MondrianForestRegressor
    >>> from shinrin.benchmark import ablation_benchmark
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(200, 4).astype(np.float32)
    >>> y = X[:, 0] * 2.0 + 1.0
    >>> variants = {
    ...     "small": MondrianForestRegressor(n_estimators=5, random_state=0),
    ...     "large": MondrianForestRegressor(n_estimators=20, random_state=0),
    ... }
    >>> results = ablation_benchmark(variants, X[:150], y[:150], X[150:], y[150:])
    >>> sorted(results)  # doctest: +SKIP
    ['large', 'small']
    """
    results = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        predict_times = []
        for _ in range(n_repeats_predict):
            start = time.perf_counter()
            model.predict(X_test)
            predict_times.append(time.perf_counter() - start)

        results[name] = {
            "fit_time": float(fit_time),
            "predict_time": float(np.mean(predict_times)),
            "train_score": float(model.score(X_train, y_train)),
            "test_score": float(model.score(X_test, y_test)),
        }
    return results


def print_ablation_report(results: dict[str, dict[str, Any]]) -> None:
    """Print an ablation table with deltas against the first variant.

    Parameters
    ----------
    results : dict
        Output from :func:`ablation_benchmark`.
    """
    names = list(results)
    if not names:
        return
    baseline = results[names[0]]

    print("=" * 72)
    print("  Ablation Report (baseline: first row)")
    print("=" * 72)
    header = f"{'variant':<20} {'fit':>10} {'Δfit':>9} {'test score':>12} {'Δscore':>9}"
    print(header)
    print("-" * 72)
    for name, m in results.items():
        d_fit = m["fit_time"] / max(baseline["fit_time"], 1e-12)
        d_score = m["test_score"] - baseline["test_score"]
        print(
            f"{name:<20} {m['fit_time']:>8.3f}s {d_fit:>8.2f}x "
            f"{m['test_score']:>12.4f} {d_score:>+9.4f}"
        )
    print("=" * 72)


def print_benchmark_report(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted benchmark report.

    Parameters
    ----------
    results : dict
        Output from ``full_benchmark``.
    """
    print("=" * 60)
    print("  Shinrin Benchmark Report")
    print("=" * 60)

    # Group by model name
    models = {}
    for key, metrics in results.items():
        model_name = key.rsplit(".", 1)[0]
        metric_type = key.rsplit(".", 1)[1]
        if model_name not in models:
            models[model_name] = {}
        models[model_name][metric_type] = metrics

    for model_name, metrics in models.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)

        if "train" in metrics:
            t = metrics["train"]
            print(f"  Training: {t['mean_time']:.4f}s (±{t['std_time']:.4f})")

        if "predict" in metrics:
            p = metrics["predict"]
            print(f"  Prediction: {p['mean_time']:.6f}s (±{p['std_time']:.6f})")
            if p.get("predictions_shape"):
                print(f"    Output shape: {p['predictions_shape']}")

        if "size" in metrics:
            s = metrics["size"]
            print(f"  Nodes: {s['n_nodes']}, Leaves: {s['n_leaves']}")
            print(f"  Estimators: {s['n_estimators']}")

    print("\n" + "=" * 60)
