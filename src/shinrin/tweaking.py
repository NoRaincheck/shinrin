"""Minimal feature tweaks that flip predictions across sets of trees.

Variation on Tolomei et al., *Interpretable Predictions of Tree-based
Ensembles via Actionable Feature Tweaking* (KDD 2017, arXiv:1706.06691),
adapted to shinrin's SPOT (optimal sparse tree) and SPOTSET (Rashomon set of
sparse optimal trees) estimators.

Tweaking applies to any tree-like object: any fitted classifier whose
prediction decomposes into root-to-leaf paths of per-feature tests.
Supported out of the box are :class:`shinrin.SPOTClassifier`,
:class:`shinrin.SPOTSETClassifier`, scikit-learn decision trees and any
forest exposing ``estimators_``; other families only need to supply leaf
literal sets plus batch prediction (see ``shinrin._tweaking._adapters``).

Two search scopes are supported:

- ``scope="reference"``: flip only the reference model (the first/optimal
  tree), the classical single-model counterfactual;
- ``scope="rashomon"``: flip *every* model in the set simultaneously — a
  tweak guaranteed to change the prediction regardless of which near-optimal
  model is deployed. Because Rashomon members share most decision structure,
  this robust tweak is typically found at little extra cost over the
  reference one; the same query posed to a decorrelated random forest is
  routinely infeasible (the search proves it or exhausts its budget).

Guarantee semantics: minimality and the verified flip hold on a **single
tree** basis. On an ensemble, a reference-scope tweak flips one member and
says nothing about the aggregated vote of the rest — single-tree tweaking
is not robust in ensembles. It is therefore ideal when the whole model is
exactly one tree, i.e. SPOT: an exact minimal counterfactual for a globally
optimal sparse tree. Robustness across many models exists only through
``scope="rashomon"``, which is exact for the given finite model set but is
a property of that enumerated set, not a general ensemble guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._tweaking._adapters import SpotsetView, SpotView
from ._tweaking._core import (
    FlipOutcome,
    build_models_target_leaves,
    reference_minimal_flip,
    robust_minimal_flip,
)


@dataclass(frozen=True)
class FlipResult:
    """Per-sample outcome of a minimal-flip search."""

    index: int
    target: Any
    scope: str
    success: bool
    optimal: bool
    verified: bool
    x_new: np.ndarray | None
    changed_features: tuple[int, ...]
    l1_distance: float
    n_models_total: int
    n_models_agree_before: int
    n_models_flipped_after: int
    nodes_expanded: int
    time_s: float


class RashomonFlipSearch:
    """Minimal-flip feature tweaking over any tree-like model or model set.

    Tweaking is generic: it works on any fitted classifier whose decisions
    decompose into root-to-leaf per-feature tests. Supported out of the box:

    - :class:`shinrin.SPOTClassifier` — a single globally optimal sparse tree;
    - :class:`shinrin.SPOTSETClassifier` — the full Rashomon set;
    - scikit-learn decision trees and forests (anything exposing ``tree_``
      or ``estimators_``, e.g. ``RandomForestClassifier``).

    Other tree families can be plugged in by providing leaf literal sets and
    batch prediction (see ``shinrin._tweaking._adapters``).

    Guarantee semantics: results are exact and verified **per individual
    tree**. With ``scope="reference"`` on an ensemble, only one member is
    flipped — there is no guarantee about the aggregated vote of the other
    members, so single-tree tweaking is *not* robust in ensembles. That makes
    it ideal for SPOT, where the deployed model is exactly one (globally
    optimal, sparse) tree and the counterfactual is minimal and verified with
    respect to the entire model. Ensemble-level robustness is available only
    through ``scope="rashomon"``, which is exact for a given finite model set;
    it is cheap on SPOTSETs but typically infeasible on decorrelated random
    forests.

    Parameters
    ----------
    estimator :
        A *fitted* :class:`shinrin.SPOTSETClassifier`,
        :class:`shinrin.SPOTClassifier`, or scikit-learn forest/tree
        classifier.

    Notes
    -----
    ``search`` expects the same feature matrix used at fit time (binarized
    columns for SPOT/SPOTSET); tweaks are reported as column indices into
    that space.
    """

    def __init__(self, estimator: Any):
        self.estimator_ = estimator
        self.classes_ = np.asarray(estimator.classes_)
        if hasattr(estimator, "model_set_"):
            self._view = SpotsetView(estimator)
        elif hasattr(estimator, "get_result"):
            self._view = SpotView(estimator)
        elif hasattr(estimator, "estimators_") or hasattr(estimator, "tree_"):
            from ._tweaking._sklearn_adapter import SklearnForestView

            self._view = SklearnForestView(estimator)
        else:
            raise TypeError(
                "Unsupported estimator: expected a fitted SPOTSETClassifier, "
                "SPOTClassifier, or scikit-learn forest/tree classifier"
            )

    def search(
        self,
        X: np.ndarray,
        *,
        target: Any | None = None,
        scope: str = "rashomon",
        max_nodes: int = 100_000,
        time_limit: float | None = None,
    ) -> list[FlipResult]:
        """Search minimal flips for every row of ``X``.

        Parameters
        ----------
        X :
            Feature matrix in the estimator's training space.
        target :
            Desired class (original label space). ``None`` means "flip" —
            any class different from the reference model's prediction;
            requires binary problems.
        scope :
            ``"rashomon"`` flips every model in the set; ``"reference"``
            flips only the first (optimal) tree. Guarantees are per
            individual tree: a reference-scope tweak of an ensemble member
            carries no robustness for the ensemble's aggregated vote.
        """
        if scope not in ("rashomon", "reference"):
            raise ValueError(f"Unknown scope {scope!r}")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self._view.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but the models were fitted "
                f"with {self._view.n_features}"
            )
        preds_before = self._view.predict_all(X)
        results: list[FlipResult] = []
        for i in range(X.shape[0]):
            results.append(
                self._search_row(
                    X,
                    preds_before,
                    i,
                    target=target,
                    scope=scope,
                    max_nodes=max_nodes,
                    time_limit=time_limit,
                )
            )
        return results

    def _resolve_target_index(
        self, ref_prediction: int, target: Any | None, index: int
    ) -> int:
        if target is None:
            others = [k for k in range(len(self.classes_)) if k != ref_prediction]
            if len(others) != 1:
                raise ValueError(
                    "Ambiguous flip target for multiclass output at row "
                    f"{index}; pass an explicit ``target``."
                )
            return others[0]
        matches = np.flatnonzero(self.classes_ == target)
        if matches.size != 1:
            raise ValueError(f"Target {target!r} not among classes {self.classes_}")
        return int(matches[0])

    def _search_row(
        self,
        X: np.ndarray,
        preds_before: np.ndarray,
        i: int,
        *,
        target: Any | None,
        scope: str,
        max_nodes: int,
        time_limit: float | None,
    ) -> FlipResult:
        x = X[i]
        target_idx = self._resolve_target_index(int(preds_before[0, i]), target, i)
        agree_before = int(np.sum(preds_before[:, i] == target_idx))

        all_model_leaves = [
            [(c, label == target_idx) for c, label in model_leaves]
            for model_leaves in self._view.leaves
        ]
        target_leaves, predicts_target = build_models_target_leaves(all_model_leaves, x)

        outcome: FlipOutcome
        if scope == "reference":
            outcome = reference_minimal_flip(target_leaves[0], x, predicts_target[0])
        else:
            outcome = robust_minimal_flip(
                target_leaves, x, max_nodes=max_nodes, time_limit=time_limit
            )

        flipped_after = 0
        verified = False
        x_new = outcome.x_new
        if outcome.success and x_new is not None:
            preds_after = self._view.predict_all(x_new.reshape(1, -1))[:, 0]
            flipped_after = int(np.sum(preds_after == target_idx))
            required = preds_after.shape[0] if scope == "rashomon" else 1
            verified = flipped_after >= required
            if not verified:
                raise RuntimeError(
                    f"Internal inconsistency at row {i}: solver reported a "
                    f"flipping tweak but verification flipped "
                    f"{flipped_after}/{required} models"
                )

        return FlipResult(
            index=i,
            target=self.classes_[target_idx],
            scope=scope,
            success=outcome.success,
            optimal=outcome.optimal,
            verified=verified,
            x_new=x_new,
            changed_features=outcome.delta_features,
            l1_distance=outcome.l1_distance,
            n_models_total=preds_before.shape[0],
            n_models_agree_before=agree_before,
            n_models_flipped_after=flipped_after,
            nodes_expanded=outcome.nodes_expanded,
            time_s=outcome.time_s,
        )


def summarize_flip_results(results: Sequence[FlipResult]) -> dict[str, Any]:
    """Aggregate statistics over a batch of flip results."""
    n = len(results)
    successes = [r for r in results if r.success]
    distances = [r.l1_distance for r in successes]
    proven_infeasible = [r for r in results if not r.success and r.optimal]
    budget_limited = [r for r in results if not r.success and not r.optimal]
    return {
        "n_samples": n,
        "success_rate": len(successes) / n if n else 0.0,
        "proven_infeasible_rate": len(proven_infeasible) / n if n else 0.0,
        "budget_exhausted_rate": len(budget_limited) / n if n else 0.0,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "median_distance": float(np.median(distances)) if distances else None,
        "max_distance": max(distances) if distances else None,
        "mean_changed_features": (
            float(np.mean([len(r.changed_features) for r in successes]))
            if successes
            else None
        ),
        "total_nodes": sum(r.nodes_expanded for r in results),
        "total_time_s": sum(r.time_s for r in results),
    }
