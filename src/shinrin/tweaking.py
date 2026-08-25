"""Minimal feature tweaks that flip predictions across sets of trees.

Variation on Tolomei et al., *Interpretable Predictions of Tree-based
Ensembles via Actionable Feature Tweaking* (KDD 2017, arXiv:1706.06691),
adapted to shinrin's SPOT (optimal sparse tree) and SPOTSET (Rashomon set of
sparse optimal trees) estimators.

Tweaking applies to any tree-like object: any fitted classifier whose
prediction decomposes into root-to-leaf paths of per-feature tests.
Supported out of the box are :class:`shinrin.SPOTClassifier`,
:class:`shinrin.SPOTSETClassifier`, scikit-learn decision trees, forests
(``RandomForestClassifier``, ``ExtraTreesClassifier``, ``BaggingClassifier``),
voting committees (``AdaBoostClassifier``, weights honoured) and boosted
ensembles (``GradientBoostingClassifier``, ``HistGradientBoostingClassifier``,
binary); other families only need to supply leaf literal sets plus batch
prediction (see ``shinrin._tweaking._adapters``).

Three search scopes are supported:

- ``scope="reference"``: flip only the reference model (the first/optimal
  tree), the classical single-model counterfactual;
- ``scope="rashomon"``: flip *every* model in the set simultaneously — a
  tweak guaranteed to change the prediction regardless of which near-optimal
  model is deployed. Because Rashomon members share most decision structure,
  this robust tweak is typically found at little extra cost over the
  reference one; the same query posed to a decorrelated random forest is
  routinely infeasible (the search proves it or exhausts its budget);
- ``scope="ensemble"``: flip the estimator's own aggregated prediction,
  using each family's actual decision rule — soft probability voting for
  forests, weighted hard votes for AdaBoost-style committees, score-sign
  crossing for boosters — verified against the real ``predict``.

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
    Choice,
    FlipOutcome,
    aggregate_minimal_flip,
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
    - scikit-learn decision trees, forests and bagging ensembles;
    - ``AdaBoostClassifier`` (estimator weights honoured in votes);
    - ``GradientBoostingClassifier`` / ``HistGradientBoostingClassifier``
      (binary; score-sign crossing over stage outputs).

    Other tree families can be plugged in by providing leaf literal sets and
    batch prediction (see ``shinrin._tweaking._adapters``).

    Guarantee semantics: results are exact and verified **per individual
    tree**. With ``scope="reference"`` on an ensemble, only one member is
    flipped — there is no guarantee about the aggregated vote of the other
    members, so single-tree tweaking is *not* robust in ensembles. That makes
    it ideal for SPOT, where the deployed model is exactly one (globally
    optimal, sparse) tree and the counterfactual is minimal and verified with
    respect to the entire model. Ensemble-level robustness is available only
    through ``scope="rashomon"`` (exact for a given finite model set; cheap
    on SPOTSETs, typically infeasible on decorrelated random forests) or
    through ``scope="ensemble"``, which guarantees the tweak flips the
    estimator's own aggregated prediction — weaker than all-members
    robustness, but always verified against the real ``predict``.

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
        # mode: "set" (label members), "committee" (hard weighted vote),
        # "proba" (soft-voting trees/forests), "boosting" (score stages).
        if hasattr(estimator, "model_set_"):
            self._view = SpotsetView(estimator)
            self._mode = "set"
        elif hasattr(estimator, "get_result"):
            self._view = SpotView(estimator)
            self._mode = "set"
        elif hasattr(estimator, "_predictors"):
            from ._tweaking._sklearn_adapter import HistGradientBoostingView

            self._view = HistGradientBoostingView(estimator)
            self._mode = "boosting"
        elif hasattr(estimator, "estimators_"):
            from ._tweaking._sklearn_adapter import (
                GradientBoostingView,
                SklearnForestView,
            )

            if isinstance(getattr(estimator, "estimators_", None), np.ndarray):
                self._view: Any = GradientBoostingView(estimator)
                self._mode = "boosting"
            elif getattr(estimator, "estimator_weights_", None) is not None:
                self._view = SklearnForestView(estimator)
                self._mode = "committee"
            else:
                self._view = SklearnForestView(estimator)
                self._mode = "proba"
        elif hasattr(estimator, "tree_"):
            from ._tweaking._sklearn_adapter import SklearnForestView

            self._view = SklearnForestView(estimator)
            self._mode = "proba"
        else:
            raise TypeError(
                "Unsupported estimator: expected a fitted tree-like "
                "classifier (SPOT, SPOTSET, sklearn tree/forest/committee/"
                "booster); other families need a leaf-literal adapter"
            )
        if self._mode == "boosting" and len(self.classes_) != 2:
            raise NotImplementedError(
                "Additive (boosted) tweaking supports binary problems only"
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
            flips only the first (optimal) tree; ``"ensemble"`` flips the
            estimator's own aggregated prediction using its real decision
            rule (soft vote / weighted vote / score crossing), verified
            against ``estimator.predict``. Guarantees for ``"reference"``
            and ``"rashomon"`` are per individual tree: a reference-scope
            tweak of an ensemble member carries no robustness for the
            ensemble's aggregated vote.
        """
        if scope not in ("rashomon", "reference", "ensemble"):
            raise ValueError(f"Unknown scope {scope!r}")
        if scope == "rashomon" and self._mode == "boosting":
            raise ValueError(
                "scope='rashomon' (flip all members) is undefined for "
                "boosted score ensembles; use scope='ensemble'"
            )
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self._view.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but the models were fitted "
                f"with {self._view.n_features}"
            )
        preds_before = self._member_predictions(X)
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

    def _member_predictions(self, X: np.ndarray) -> np.ndarray:
        """Encoded per-member predictions, shape ``(n_members, n_samples)``."""
        if self._mode == "boosting":
            return (self._view.score_matrix(X) > 0).astype(int)
        return self._view.predict_all(X)

    def _ensemble_engine(
        self,
        x: np.ndarray,
        target_idx: int,
        max_nodes: int,
        time_limit: float | None,
    ) -> FlipOutcome:
        """Minimal tweak flipping the estimator's own aggregated prediction."""
        if self._mode == "set":
            all_model_leaves = [
                [(c, label == target_idx) for c, label in model_leaves]
                for model_leaves in self._view.leaves
            ]
            target_leaves, _ = build_models_target_leaves(all_model_leaves, x)
            return robust_minimal_flip(
                target_leaves,
                x,
                max_nodes=max_nodes,
                time_limit=time_limit,
            )
        if self._mode == "committee":
            weights = self._view.weights
            choices = []
            for m, model_leaves in enumerate(self._view.leaves):
                weight = 1.0 if weights is None else float(weights[m])
                options = [
                    Choice(c, weight)
                    for c, label in model_leaves
                    if label == target_idx
                ]
                options.append(Choice({}, 0.0))
                choices.append(options)
            total = (
                float(weights.sum())
                if weights is not None
                else float(len(self._view.leaves))
            )
            return aggregate_minimal_flip(
                choices,
                x,
                threshold=total / 2.0,
                strict=True,
                max_nodes=max_nodes,
                time_limit=time_limit,
            )
        if self._mode == "proba":
            class_index = int(
                np.flatnonzero(self.classes_ == self.classes_[target_idx])[0]
            )
            proba_leaves = self._view.probability_leaf_values(class_index)
            n_members = len(proba_leaves)
            already = (
                self.estimator_.predict(x.reshape(1, -1))[0]
                == self.classes_[target_idx]
            )
            if already:
                return FlipOutcome(
                    True, np.array(x, dtype=float), (), 0.0, 0, True, 0.0
                )
            choices = [[Choice(c, p) for c, p in tree] for tree in proba_leaves]
            return aggregate_minimal_flip(
                choices,
                x,
                threshold=n_members / 2.0,
                strict=True,
                max_nodes=max_nodes,
                time_limit=time_limit,
            )
        # boosting: score-sign crossing (binary)
        sign = 1.0 if target_idx == 1 else -1.0
        base = sign * self._view.base_score
        already = (
            self.estimator_.predict(x.reshape(1, -1))[0] == self.classes_[target_idx]
        )
        if already:
            return FlipOutcome(True, np.array(x, dtype=float), (), 0.0, 0, True, 0.0)
        choices = [
            [Choice(c, sign * self._view.scale * v) for c, v in stage]
            for stage in self._view.leaf_values
        ]
        return aggregate_minimal_flip(
            choices,
            x,
            base=base,
            strict=True,
            max_nodes=max_nodes,
            time_limit=time_limit,
        )

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

        outcome: FlipOutcome
        if scope == "ensemble" and self._mode != "set":
            outcome = self._ensemble_engine(x, target_idx, max_nodes, time_limit)
        elif scope == "reference" and self._mode == "boosting":
            outcome = self._boosting_reference(x, target_idx, max_nodes, time_limit)
        else:
            all_model_leaves = [
                [(c, label == target_idx) for c, label in model_leaves]
                for model_leaves in self._view.leaves
            ]
            target_leaves, predicts_target = build_models_target_leaves(
                all_model_leaves, x
            )
            if scope == "reference":
                outcome = reference_minimal_flip(
                    target_leaves[0], x, predicts_target[0]
                )
            else:
                # rashomon scope; ensemble on a set means the same all-flip
                outcome = robust_minimal_flip(
                    target_leaves,
                    x,
                    max_nodes=max_nodes,
                    time_limit=time_limit,
                )

        flipped_after = 0
        verified = False
        x_new = outcome.x_new
        if outcome.success and x_new is not None:
            flipped_after = self._post_tweak_support(x_new, target_idx)
            if scope in ("rashomon", "ensemble") and self._mode == "set":
                verified = flipped_after >= preds_before.shape[0]
            elif scope == "reference":
                verified = flipped_after >= 1
            else:
                # ensemble scopes verify against the estimator's real predict
                prediction = self.estimator_.predict(x_new.reshape(1, -1))[0]
                verified = bool(prediction == self.classes_[target_idx])
            if not verified:
                raise RuntimeError(
                    f"Internal inconsistency at row {i}: solver reported a "
                    f"flipping tweak but verification failed "
                    f"({flipped_after} members support the target)"
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

    def _boosting_reference(
        self,
        x: np.ndarray,
        target_idx: int,
        max_nodes: int,
        time_limit: float | None,
    ) -> FlipOutcome:
        """Reference scope on a booster: first-stage score crossing."""
        sign = 1.0 if target_idx == 1 else -1.0
        stage = self._view.leaf_values[0]
        choices = [[Choice(c, sign * self._view.scale * v) for c, v in stage]]
        return aggregate_minimal_flip(
            choices,
            x,
            base=sign * self._view.base_score,
            strict=True,
            max_nodes=max_nodes,
            time_limit=time_limit,
        )

    def _post_tweak_support(self, x_new: np.ndarray, target_idx: int) -> int:
        """Members/stages supporting the target after the tweak."""
        if self._mode == "boosting":
            signs = self._view.score_matrix(x_new.reshape(1, -1))[:, 0]
            favorable = signs > 0 if target_idx == 1 else signs < 0
            return int(np.sum(favorable))
        preds_after = self._view.predict_all(x_new.reshape(1, -1))[:, 0]
        return int(np.sum(preds_after == target_idx))


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
