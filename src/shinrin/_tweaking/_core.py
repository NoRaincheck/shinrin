"""Core minimal-flip search over sets of tree models.

A model (decision tree) is abstracted to the list of root-to-leaf literal
sets that lead to a desired target class. Each literal constrains one
feature to an inclusive interval ``[lo, hi]``:

- binary equality ``x_f == v``      ->  ``[v, v]``
- threshold ``x_f <= t``            ->  ``[-inf, t]``
- threshold ``x_f > t``             ->  ``[t + eps, +inf]``

A point satisfying every interval of a leaf's literal set provably traverses
the tree to that leaf. Flipping a *set* of models therefore reduces to
choosing one target leaf per model and intersecting the chosen literal sets;
the L1 distance from ``x`` to the intersection is the tweak cost, and the
minimum over choices is the minimal flip.

The exact search is best-first (A*) over partial leaf combinations with an
admissible suffix lower bound. Because the marginal cost of adding a leaf to
an existing intersection can drop below that leaf's standalone cost, the
heuristic is not consistent; states are therefore reopened when a cheaper
path is found, preserving optimality. Node/time budgets degrade gracefully
to "best found, optimality unproven", which is how deliberately hard or
unsolvable ensembles (e.g. large random forests) are handled.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Interval = tuple[float, float]
Constraint = dict[int, Interval]

INF = float("inf")


@dataclass(frozen=True)
class Choice:
    """One candidate decision for a model in the aggregate search.

    ``constraint`` is the leaf's literal set; ``contribution`` is what the
    model adds to the aggregate when this choice is taken — a vote weight
    (0 for a free "not flipped" choice) or an additive score term.
    """

    constraint: Constraint
    contribution: float


def merge_constraints(a: Constraint, b: Constraint) -> Constraint | None:
    """Intersect two literal sets; return None if they conflict on a feature."""
    out = dict(a)
    for f, (lo_b, hi_b) in b.items():
        lo_a, hi_a = out.get(f, (-INF, INF))
        lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
        if lo > hi:
            return None
        out[f] = (lo, hi)
    return out


def satisfies(constraint: Constraint, x: np.ndarray) -> bool:
    """Whether point ``x`` satisfies every interval of the literal set."""
    return all(lo <= x[f] <= hi for f, (lo, hi) in constraint.items())


def interval_distance(value: float, interval: Interval) -> float:
    lo, hi = interval
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def constraint_cost(constraint: Constraint, x: np.ndarray) -> float:
    """L1 distance from ``x`` to the region satisfying the literal set."""
    return sum(interval_distance(x[f], iv) for f, iv in constraint.items())


def changed_features(constraint: Constraint, x: np.ndarray) -> tuple[int, ...]:
    """Features whose value the literal set forces away from ``x``."""
    return tuple(
        sorted(f for f, iv in constraint.items() if interval_distance(x[f], iv) > 0.0)
    )


def project(x: np.ndarray, constraint: Constraint) -> np.ndarray:
    """Nearest point to ``x`` inside the literal-set region (L1 projection)."""
    out = np.array(x, dtype=float)
    for f, (lo, hi) in constraint.items():
        out[f] = min(max(out[f], lo), hi)
    return out


@dataclass(frozen=True)
class FlipOutcome:
    """Result of a minimal-flip search for one sample."""

    success: bool
    x_new: np.ndarray | None
    delta_features: tuple[int, ...]
    l1_distance: float
    nodes_expanded: int
    optimal: bool
    time_s: float


@dataclass(order=True)
class _Entry:
    priority: float
    tie_breaker: int
    model_idx: int = field(compare=False)
    constraint_idx: int = field(compare=False)


class _RobustFlipSearcher:
    def __init__(
        self,
        models_target_leaves: list[list[Constraint]],
        x: np.ndarray,
        max_nodes: int,
        time_limit: float | None,
    ):
        self.leaves = models_target_leaves
        self.x = x
        self.n_models = len(models_target_leaves)
        self.max_nodes = max_nodes
        self.deadline = None if time_limit is None else time.monotonic() + time_limit
        self.nodes_expanded = 0

        # Registry of constructed intersections; index 0 is the empty set.
        self.constraints: list[Constraint] = [{}]

        # Admissible suffix bound: any tweak flipping model i costs at least
        # its cheapest single-model flip. Note a model that already predicts
        # the target still contributes its satisfied leaf (cost 0 here) — the
        # joint tweak must be prevented from knocking it off that leaf.
        self.suffix_min = [0.0] * (self.n_models + 1)
        for i in range(self.n_models - 1, -1, -1):
            rest = self.suffix_min[i + 1]
            if self.leaves[i]:
                own = min(constraint_cost(c, x) for c in self.leaves[i])
            else:
                own = INF
            self.suffix_min[i] = INF if own == INF or rest == INF else rest + own

    def run(self) -> FlipOutcome:
        start = time.monotonic()
        counter = itertools.count()
        heap: list[_Entry] = [_Entry(self.suffix_min[0], next(counter), 0, 0)]
        # Canonical states: (model layer, frozen intersection) -> constraint
        # registry index; best_g tracks the cheapest known reachability cost
        # and allows reopening when the heuristic proves inconsistent.
        registry: dict[tuple[int, frozenset], int] = {(0, frozenset()): 0}
        best_g: dict[tuple[int, frozenset], float] = {(0, frozenset()): 0.0}
        solution: tuple[float, Constraint] | None = None
        exhausted = False

        while heap:
            if self.deadline is not None and time.monotonic() > self.deadline:
                exhausted = True
                break
            entry = heapq.heappop(heap)
            i, ci = entry.model_idx, entry.constraint_idx
            merged = self.constraints[ci]
            canon = (i, frozenset(merged.items()))
            g_here = constraint_cost(merged, self.x)
            if best_g.get(canon, INF) < g_here:
                continue

            if i == self.n_models:
                solution = (g_here, merged)
                break
            self.nodes_expanded += 1
            if self.nodes_expanded > self.max_nodes:
                exhausted = True
                break

            choices = self.leaves[i]
            for choice in choices:
                new_c = merge_constraints(merged, choice)
                if new_c is None or self.suffix_min[i + 1] == INF:
                    continue
                new_g = constraint_cost(new_c, self.x)
                new_canon = (i + 1, frozenset(new_c.items()))
                prior = best_g.get(new_canon)
                if prior is not None and prior <= new_g:
                    continue
                idx = registry.get(new_canon)
                if idx is None:
                    self.constraints.append(new_c)
                    idx = len(self.constraints) - 1
                    registry[new_canon] = idx
                best_g[new_canon] = new_g
                heapq.heappush(
                    heap,
                    _Entry(
                        new_g + self.suffix_min[i + 1],
                        next(counter),
                        i + 1,
                        idx,
                    ),
                )

        elapsed = time.monotonic() - start
        if solution is not None:
            g, merged = solution
            x_new = project(self.x, merged)
            return FlipOutcome(
                success=True,
                x_new=x_new,
                delta_features=changed_features(merged, self.x),
                l1_distance=g,
                nodes_expanded=self.nodes_expanded,
                optimal=not exhausted,
                time_s=elapsed,
            )
        return FlipOutcome(
            success=False,
            x_new=None,
            delta_features=(),
            l1_distance=INF,
            nodes_expanded=self.nodes_expanded,
            optimal=not exhausted,
            time_s=elapsed,
        )


class _AggregateFlipSearcher:
    """Uniform-cost search over per-model choices with an aggregate goal.

    Minimizes the L1 cost of the merged literal sets subject to
    ``base + sum(contributions) >/>= threshold`` at completion. Used for
    weighted-majority committees (contributions are vote weights, free
    "not flipped" choices carry 0) and boosted score ensembles (every tree
    contributes its leaf value; no free choice). Exact: states expand in
    cost order and Pareto-dominance dedup never discards a cheaper or
    higher-aggregate alternative.
    """

    def __init__(
        self,
        model_choices: list[list[Choice]],
        x: np.ndarray,
        base: float,
        threshold: float,
        strict: bool,
        max_nodes: int,
        time_limit: float | None,
    ):
        self.choices = model_choices
        self.x = x
        self.n_models = len(model_choices)
        self.base = base
        self.threshold = threshold
        self.strict = strict
        self.max_nodes = max_nodes
        self.deadline = None if time_limit is None else time.monotonic() + time_limit
        self.nodes_expanded = 0

        self.constraints: list[Constraint] = [{}]
        # Suffix upper bounds on achievable aggregate gain, for pruning.
        self.suffix_gain = [0.0] * (self.n_models + 1)
        for i in range(self.n_models - 1, -1, -1):
            best = max((c.contribution for c in self.choices[i]), default=-INF)
            rest = self.suffix_gain[i + 1]
            self.suffix_gain[i] = (
                INF if (rest == INF or best == -INF) else rest + max(best, 0.0)
            )
        self._can_reach: Callable[[float, int], bool] = (
            self._reach_strict if strict else self._reach_nonstrict
        )

    def _reach_strict(self, agg: float, i: int) -> bool:
        bound = self.suffix_gain[i]
        return bound != INF and agg + bound > self.threshold

    def _reach_nonstrict(self, agg: float, i: int) -> bool:
        bound = self.suffix_gain[i]
        return bound != INF and agg + bound >= self.threshold

    def run(self) -> FlipOutcome:
        start = time.monotonic()
        counter = itertools.count()
        heap: list[_Entry] = [_Entry(0.0, next(counter), 0, 0)]
        # Pareto frontier of (cost, aggregate) per canonical state.
        frontier: dict[tuple[int, frozenset], list[tuple[float, float]]] = {
            (0, frozenset()): [(0.0, self.base)]
        }
        solution: tuple[float, Constraint] | None = None
        exhausted = False

        while heap:
            if self.deadline is not None and time.monotonic() > self.deadline:
                exhausted = True
                break
            entry = heapq.heappop(heap)
            i, ci = entry.model_idx, entry.constraint_idx
            merged = self.constraints[ci]
            canon = (i, frozenset(merged.items()))
            g_here = constraint_cost(merged, self.x)
            live = [
                (g, agg)
                for g, agg in frontier[canon]
                if g <= g_here + 1e-15
            ]
            if not live:
                continue
            # Highest aggregate among cheapest-known entries.
            agg_here = max(agg for _, agg in live)

            if i == self.n_models:
                if self._can_reach(agg_here, i):
                    solution = (g_here, merged)
                    break
                continue
            self.nodes_expanded += 1
            if self.nodes_expanded > self.max_nodes:
                exhausted = True
                break

            for choice in self.choices[i]:
                new_c = merge_constraints(merged, choice.constraint)
                if new_c is None or not self._can_reach(
                    agg_here + choice.contribution, i + 1
                ):
                    continue
                new_g = constraint_cost(new_c, self.x)
                new_canon = (i + 1, frozenset(new_c.items()))
                entries = frontier.setdefault(new_canon, [])
                if any(
                    g <= new_g + 1e-15 and agg >= agg_here + choice.contribution
                    for g, agg in entries
                ):
                    continue
                entries[:] = [
                    (g, agg)
                    for g, agg in entries
                    if not (
                        new_g <= g + 1e-15
                        and agg_here + choice.contribution >= agg
                    )
                ]
                entries.append((new_g, agg_here + choice.contribution))
                idx = len(self.constraints)
                self.constraints.append(new_c)
                heapq.heappush(
                    heap, _Entry(new_g, next(counter), i + 1, idx)
                )

        elapsed = time.monotonic() - start
        if solution is not None:
            g, merged = solution
            return FlipOutcome(
                success=True,
                x_new=project(self.x, merged),
                delta_features=changed_features(merged, self.x),
                l1_distance=g,
                nodes_expanded=self.nodes_expanded,
                optimal=not exhausted,
                time_s=elapsed,
            )
        return FlipOutcome(
            success=False,
            x_new=None,
            delta_features=(),
            l1_distance=INF,
            nodes_expanded=self.nodes_expanded,
            optimal=not exhausted,
            time_s=elapsed,
        )


def aggregate_minimal_flip(
    model_choices: list[list[Choice]],
    x: np.ndarray,
    *,
    base: float = 0.0,
    threshold: float = 0.0,
    strict: bool = True,
    max_nodes: int = 100_000,
    time_limit: float | None = None,
) -> FlipOutcome:
    """Minimal L1 tweak whose per-model contributions cross a threshold.

    Each model must take exactly one :class:`Choice`; the search finds the
    cheapest combination of compatible leaf constraints whose aggregated
    contribution satisfies ``base + total >/>= threshold``.
    """
    searcher = _AggregateFlipSearcher(
        model_choices, x, base, threshold, strict, max_nodes, time_limit
    )
    return searcher.run()


def build_models_target_leaves(
    all_model_leaves: list[list[tuple[Constraint, bool]]],
    x: np.ndarray,
) -> tuple[list[list[Constraint]], list[bool]]:
    """Reduce full leaf inventories to target-class leaves per model.

    ``all_model_leaves[m]`` holds ``(constraint, label_is_target)`` pairs.
    The returned per-model flags report whether the currently satisfied leaf
    carries a target label (useful for reporting and for single-model flips);
    the robust search itself always requires one target leaf per model,
    including the currently satisfied one, so the joint tweak cannot silently
    knock an agreeing model off its decision path.
    """
    target_leaves: list[list[Constraint]] = []
    predicts: list[bool] = []
    for leaves in all_model_leaves:
        target_leaves.append([c for c, is_target in leaves if is_target])
        current_is_target = False
        for c, is_target in leaves:
            if satisfies(c, x):
                current_is_target = is_target
                break
        predicts.append(current_is_target)
    return target_leaves, predicts


def robust_minimal_flip(
    models_target_leaves: list[list[Constraint]],
    x: np.ndarray,
    *,
    max_nodes: int = 100_000,
    time_limit: float | None = None,
) -> FlipOutcome:
    """Minimal L1 tweak of ``x`` making every model reach a target leaf.

    Parameters
    ----------
    models_target_leaves :
        Per model, the literal sets of root-to-leaf paths labelled with the
        desired target class. Every model must contribute one chosen leaf —
        for a model already predicting the target its satisfied leaf is
        simply a zero-cost choice that pins the joint tweak to keep it there.
    x :
        Original binary/continuous feature vector.

    Returns
    -------
    FlipOutcome
        ``success=False`` with ``optimal=True`` certifies that no tweak can
        flip every model (proven infeasible). Budget exhaustion yields
        ``optimal=False``.
    """
    searcher = _RobustFlipSearcher(models_target_leaves, x, max_nodes, time_limit)
    return searcher.run()


def reference_minimal_flip(
    model_target_leaves: list[Constraint],
    x: np.ndarray,
    current_predicts_target: bool,
) -> FlipOutcome:
    """Exact minimal flip for a single model (closed form over its leaves)."""
    start = time.monotonic()
    if current_predicts_target:
        return FlipOutcome(True, np.array(x, dtype=float), (), 0.0, 0, True, 0.0)
    best_cost, best_leaf = INF, None
    for leaf in model_target_leaves:
        cost = constraint_cost(leaf, x)
        if cost < best_cost:
            best_cost, best_leaf = cost, leaf
    elapsed = time.monotonic() - start
    if best_leaf is None:
        return FlipOutcome(
            False, None, (), INF, len(model_target_leaves), True, elapsed
        )
    return FlipOutcome(
        success=True,
        x_new=project(x, best_leaf),
        delta_features=changed_features(best_leaf, x),
        l1_distance=best_cost,
        nodes_expanded=len(model_target_leaves),
        optimal=True,
        time_s=elapsed,
    )


def brute_force_min_flips(
    candidate_features: list[int],
    flips_all: Callable[[np.ndarray], bool],
    x: np.ndarray,
    max_size: int = 4,
) -> float:
    """Exhaustive reference solver for binary instances (used by tests).

    Toggles subsets of ``candidate_features`` in increasing size and returns
    the cardinality of the smallest subset for which ``flips_all`` holds, or
    ``inf`` if none within ``max_size``.
    """
    for size in range(max_size + 1):
        for combo in itertools.combinations(candidate_features, size):
            xc = np.array(x, dtype=float)
            for f in combo:
                xc[f] = 1.0 - xc[f]
            if flips_all(xc):
                return float(size)
    return INF
