"""Unit tests for the minimal-flip core solver (shinrin._tweaking._core).

The A* robust flip is validated against an exhaustive subset-enumeration
reference on randomized binary instances, plus hand-built continuous
interval cases.
"""

import numpy as np
import pytest

from shinrin._tweaking._core import (
    INF,
    constraint_cost,
    merge_constraints,
    project,
    robust_minimal_flip,
    satisfies,
    reference_minimal_flip,
    brute_force_min_flips,
)


def test_merge_intersects_and_detects_conflicts():
    a = {0: (-np.inf, 2.0), 1: (1.0, 1.0)}
    b = {0: (1.5, np.inf), 2: (0.0, 0.0)}
    merged = merge_constraints(a, b)
    assert merged == {0: (1.5, 2.0), 1: (1.0, 1.0), 2: (0.0, 0.0)}

    c = {0: (3.0, np.inf)}
    assert merge_constraints(a, c) is None


def test_cost_and_projection_continuous():
    x = np.array([0.0, 5.0])
    constraint = {0: (1.0, 2.0), 1: (-np.inf, 2.0)}
    assert constraint_cost(constraint, x) == pytest.approx(1.0 + 3.0)
    x_new = project(x, constraint)
    assert satisfies(constraint, x_new)
    assert x_new[0] == pytest.approx(1.0)
    assert x_new[1] == pytest.approx(2.0)


def test_reference_flip_single_model():
    x = np.array([0.0, 1.0, 0.0])
    target_leaves = [
        {0: (1.0, 1.0)},
        {1: (0.0, 0.0), 2: (1.0, 1.0)},
    ]
    outcome = reference_minimal_flip(target_leaves, x, current_predicts_target=False)
    assert outcome.success and outcome.optimal
    assert outcome.l1_distance == 1.0
    assert outcome.delta_features == (0,)
    # second leaf would cost 2 (flip feature 1 off, feature 2 on)
    closer = reference_minimal_flip(
        [{1: (0.0, 0.0), 2: (1.0, 1.0)}], x, current_predicts_target=False
    )
    assert closer.l1_distance == 2.0


def _random_instance(rng):
    n_features = int(rng.integers(4, 8))
    n_models = int(rng.integers(2, 5))
    models = []
    for _ in range(n_models):
        leaves = []
        for _ in range(int(rng.integers(1, 4))):
            size = int(rng.integers(1, min(4, n_features) + 1))
            feats = rng.choice(n_features, size=size, replace=False)
            leaves.append({int(f): (float(rng.integers(0, 2)),) * 2 for f in feats})
        models.append(leaves)
    x = rng.integers(0, 2, size=n_features).astype(float)
    return models, x


def test_robust_flip_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(42)
    for trial in range(60):
        models, x = _random_instance(rng)

        def flips_all(xc, models=models):
            return all(
                any(satisfies(leaf, xc) for leaf in model) for model in models
            )

        outcome = robust_minimal_flip(models, x)
        expected = brute_force_min_flips(list(range(x.size)), flips_all, x, max_size=4)

        if expected == INF:
            assert not outcome.success
            assert outcome.optimal, f"trial {trial}: infeasibility not proven"
        else:
            assert outcome.success, (
                f"trial {trial}: solver missed feasible flip of cost {expected}"
            )
            assert outcome.l1_distance == pytest.approx(expected), (
                f"trial {trial}: suboptimal {outcome.l1_distance} vs {expected}"
            )
            assert satisfies_any(outcome.x_new, models)
            assert len(outcome.delta_features) == int(outcome.l1_distance)


def satisfies_any(x, models):
    return all(any(satisfies(leaf, x) for leaf in model) for model in models)


def test_provably_infeasible_when_models_contradict():
    models = [
        [{0: (1.0, 1.0)}],
        [{0: (0.0, 0.0), 1: (1.0, 1.0)}],
    ]
    x = np.array([0.0, 0.0])
    outcome = robust_minimal_flip(models, x)
    assert not outcome.success
    assert outcome.optimal


def test_budget_exhaustion_degrades_gracefully():
    rng = np.random.default_rng(3)
    models, x = _random_instance(rng)
    outcome = robust_minimal_flip(models, x, max_nodes=1)
    assert not outcome.optimal
    outcome = robust_minimal_flip(models, x, time_limit=0.0)
    assert not outcome.success and not outcome.optimal


def test_agreeing_model_pins_joint_tweak():
    # Model 1 already predicts the target via {0: 1}; model 2 needs {1: 1}.
    # The joint tweak must keep model 1 on its leaf, i.e. f0 stays untouched.
    x = np.array([1.0, 0.0])
    models = [
        [{0: (1.0, 1.0)}],
        [{1: (1.0, 1.0)}],
    ]
    outcome = robust_minimal_flip(models, x)
    assert outcome.success
    assert outcome.l1_distance == 1.0
    assert outcome.delta_features == (1,)
    assert np.array_equal(outcome.x_new, [1.0, 1.0])

    # Now the agreeing model sits on {0: 0} while the other needs {0: 1}:
    # no joint flip exists and the infeasibility is proven.
    conflicting = [
        [{0: (0.0, 0.0)}],
        [{0: (1.0, 1.0)}],
    ]
    outcome = robust_minimal_flip(conflicting, np.array([0.0, 0.0]))
    assert not outcome.success
    assert outcome.optimal


def test_inf_constant():
    assert INF == float("inf")
