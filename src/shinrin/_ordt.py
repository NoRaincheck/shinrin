"""ORDT: Optimal Rule-sets from Decision Trees.

A variant of :class:`shinrin.SkopeRules` that replaces the heuristic
precision-weighted vote with CORELS' certified-optimal selection and
ordering:

1. mine    - skope-rules harvests root-to-leaf threshold conjunctions from
   bagged classification/regression trees, scored by out-of-bag
   precision/recall. Near-duplicates are deliberately kept: CORELS, not a
   similarity heuristic, decides what earns a slot.
2. select  - every surviving rule becomes one binary column of a capture
   matrix ``Z`` (``Z[i, j] = 1`` iff rule *j* fires on sample *i*); the
   vendored :class:`shinrin.CorelsClassifier` is fit with ``max_card=1``, so
   each antecedent of the learned list is exactly one mined rule. The result
   is the provably best ordered rule set *over the mined pool*.

Requires pandas (inherited from skope-rules' DataFrame-based rule scoring):
``pip install shinrin[pandas]``.
"""

from __future__ import annotations

import re
import time

import numpy as np

from shinrin._corels.corels import CorelsClassifier
from shinrin._skrules.skope_rules import SkopeRules

__all__ = ["OrdtClassifier"]

_TERM_RE = re.compile(r"^(\S+) (<=|>=|<|>|==) (\S+)$")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _feat_index(name: str) -> int:
    """Column index from a feature token ("x12" or skope-internal "__C__12")."""
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"cannot extract column index from {name!r}")
    return int(match.group(1))


def _rule_mask(rule: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a skope-rules query string ("x3 <= 2.5 and x0 > 1") on X."""
    mask = np.ones(X.shape[0], dtype=bool)
    for term in rule.split(" and "):
        match = _TERM_RE.match(term.strip())
        if match is None:
            raise ValueError(f"unparseable rule term: {term!r} in {rule!r}")
        name, op, rhs = match.groups()
        lhs = X[:, _feat_index(name)]
        if op == "==":
            if not _is_number(rhs):  # degenerate "c == c" always-true rule
                continue
            mask &= lhs == float(rhs)
        elif op == "<=":
            mask &= lhs <= float(rhs)
        elif op == ">":
            mask &= lhs > float(rhs)
        elif op == "<":
            mask &= lhs < float(rhs)
        else:  # >=
            mask &= lhs >= float(rhs)
    return mask


def _pretty_rule(rule: str) -> str:
    """Rewrite skope-internal "__C__j" feature names as "xj"."""
    return re.sub(r"\b__C__(\d+)\b", r"x\1", rule)


class OrdtClassifier(SkopeRules):
    """Optimal Rule-sets from Decision Trees (skope mining -> CORELS select).

    Inherits all rule-mining parameters from :class:`~shinrin.SkopeRules`
    (``n_estimators``, ``max_depth``, ``precision_min``, ``recall_min``,
    ``max_samples``, ...) and adds the selection parameters of the CORELS
    stage. Semantic deduplication is disabled: candidate near-duplicates are
    kept and CORELS selects among them optimally.

    Attributes
    ----------
    corels_ : CorelsClassifier
        Fitted selector over the capture matrix.
    pool_labels_ / pool_rules_ : list of str
        Candidate rules kept for selection (pretty / internal query form),
        aligned with the capture-matrix columns.
    stats_ : dict
        Pool sizes: {"mined", "usable", "selected"}.
    mine_s_, select_s_ : float
        Wall-clock seconds of the last ``fit``'s two stages.

    Examples
    --------
    >>> from shinrin import OrdtClassifier
    >>> clf = OrdtClassifier(n_estimators=10, max_depth=3, random_state=0)
    >>> clf.fit(X, y)                     # doctest: +SKIP
    >>> clf.predict(X)                    # doctest: +SKIP
    >>> clf.list_rules()                  # doctest: +SKIP
    [('x23 <= 884.5 and x27 <= 0.13', True), ('NOT x5 > 2.0', False)]
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | list[int] | None = 3,
        precision_min: float = 0.5,
        recall_min: float = 0.01,
        max_samples: float = 0.8,
        max_samples_features: float = 1.0,
        bootstrap: bool = False,
        bootstrap_features: bool = False,
        max_features: float | str | None = 1.0,
        min_samples_split: float = 2,
        max_rules: int = 150,
        c: float = 0.01,
        min_support: float = 0.01,
        n_iter: int = 10000,
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.max_rules = max_rules
        self.c = c
        self.min_support = min_support
        self.n_iter = n_iter
        super().__init__(
            feature_names=None,
            precision_min=precision_min,
            recall_min=recall_min,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_samples_features=max_samples_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            max_depth=max_depth,
            # keep near-duplicates: CORELS, not similarity heuristics,
            # decides which pool members earn a slot in the final list
            max_depth_duplication=None,
            max_features=max_features,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y, sample_weight=None) -> OrdtClassifier:
        """Mine candidate rules with skope-rules, then select optimally."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(
                f"ORDT requires binary targets in {{0, 1}}, got {set(np.unique(y))}"
            )

        # --- stage 1: mine candidate rules (parent skope machinery) --------
        t0 = time.perf_counter()
        super().fit(np.ascontiguousarray(X), y.astype(np.int64), sample_weight)
        raw_pool = self.rules_without_feature_names_
        pretty_pool = self.rules_
        mine_s = time.perf_counter() - t0
        n_raw = len(raw_pool)

        # --- stage 2: build capture matrix and select optimally -------------
        masks: list[np.ndarray] = []
        f1s: list[float] = []
        labels: list[str] = []
        rules: list[str] = []
        for (rule_str, scores), (pretty_str, _) in zip(raw_pool, pretty_pool):
            prec, rec = scores[0], scores[1]
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            mask = _rule_mask(rule_str, X)
            if not mask.any() or mask.all():
                continue  # degenerate columns carry no information
            masks.append(mask)
            labels.append(_pretty_rule(pretty_str))
            rules.append(rule_str)
        n_valid = len(masks)

        if n_valid == 0:
            raise RuntimeError(
                "ORDT mined no usable rules; relax precision_min/recall_min"
            )

        stacked = np.stack(masks)  # (n_candidates, n_samples)
        order = np.argsort(-np.asarray(f1s), kind="stable")
        sorted_rows = stacked[order].astype(np.uint8)
        # first occurrence in F1-desc order keeps the best-scoring duplicate
        _, first = np.unique(sorted_rows, axis=0, return_index=True)
        keep = order[np.sort(first)][: self.max_rules]

        Z = np.ascontiguousarray(stacked[keep].T.astype(np.uint8))
        self.pool_labels_ = [labels[i] for i in keep]
        self.pool_rules_ = [rules[i] for i in keep]
        self.stats_ = {"mined": n_raw, "usable": n_valid, "selected": len(keep)}

        corels = CorelsClassifier(
            c=self.c,
            min_support=self.min_support,
            max_card=1,
            n_iter=self.n_iter,
            verbosity=[],
        )
        t0 = time.perf_counter()
        corels.fit(Z, y.astype(np.uint8), features=self.pool_labels_)
        select_s = time.perf_counter() - t0

        self.corels_ = corels
        self.mine_s_ = mine_s
        self.select_s_ = select_s
        return self

    def predict(self, X) -> np.ndarray:
        """Predict via first-match evaluation of the optimal rule list."""
        X = np.asarray(X, dtype=np.float64)
        Z = np.stack([_rule_mask(r, X) for r in self.pool_rules_], axis=1).astype(
            np.uint8
        )
        return self.corels_.predict(Z)

    def score(self, X, y) -> float:
        """Mean accuracy on the given data and labels."""
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, self.predict(X)))

    def list_rules(self) -> list[tuple[str, bool]]:
        """Readable ordered rule list as ``(label, prediction)`` pairs.

        Negated pool members are prefixed with ``NOT ``.
        """
        out = []
        for entry in self.corels_.rl().rules:
            ants = entry["antecedents"]
            if not ants:
                continue  # default catch-all clause
            idx = ants[0]
            label = self.pool_labels_[abs(idx) - 1]
            out.append(
                (("NOT " + label) if idx < 0 else label, bool(entry["prediction"]))
            )
        return out

    def complexity(self) -> tuple[int, int]:
        """Return ``(total clauses, number of rules)`` of the learned list."""
        entries = [r for r in self.corels_.rl().rules if r["antecedents"]]
        return sum(len(r["antecedents"]) for r in entries), len(entries)
