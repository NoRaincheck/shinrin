#!/usr/bin/env python3
"""Benchmark ORDT: Optimal Rule-sets from Decision Trees.

ORDT is a two-stage extension of skope-rules that combines rule mining from
decision-tree ensembles with CORELS' certified-optimal selection:

1. mine   : SkopeRules harvests high-precision threshold-conjunction rules
            (root-to-leaf paths) from bagged classification/regression trees,
            scored by out-of-bag precision/recall. Unlike stock skope-rules,
            near-duplicate rules are kept - CORELS decides what matters.
2. select : every surviving rule becomes one binary column of a "capture
            matrix" Z (Z[i, j] = 1 iff rule j fires on sample i). CORELS is
            fit with max_card=1, so each antecedent of the optimal rule list
            is exactly one mined rule. The result is the provably best
            ordered rule set *over the mined pool*.

Models compared per dataset:

- cart   : sklearn DecisionTreeClassifier (raw features)
- skope  : vendored shinrin SkopeRules alone (weighted vote of rules)
- corels : vendored CorelsClassifier on quantile one-hot binarized features
           (its own mining; compas skips binarization - already binary)
- ordt   : the hybrid above (works directly on raw features; tree thresholds
           replace external discretization)

Requires the pandas extra for SkopeRules:
    uv run --extra pandas python scripts/benchmarks/bench_ordt.py [--repeats N]

Usage:
    python scripts/benchmarks/bench_ordt.py [--repeats N] [--max-rules K]
"""

from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

try:
    import pandas  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "bench_ordt requires the optional pandas dependency "
        "(SkopeRules evaluates rules with DataFrame queries). "
        "Run: uv run --extra pandas python scripts/benchmarks/bench_ordt.py"
    ) from exc

from shinrin import CorelsClassifier
from shinrin._corels import load_from_csv
from shinrin.rules import SkopeRules

_TERM_RE = re.compile(r"^(\S+) (<=|>=|<|>|==) (\S+)$")


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


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


class OrdtClassifier:
    """Optimal Rule-set from Decision Trees (skope mining -> CORELS select).

    Parameters mirror the two stages: ``n_estimators``/``max_depth``/
    ``precision_min``/``recall_min``/``max_samples`` feed SkopeRules;
    ``max_rules`` caps the pool handed to CORELS (top by OOB F1 after
    deduplicating identical capture sets); ``c``/``min_support``/``n_iter``
    configure CORELS with ``max_card=1``.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 3,
        precision_min: float = 0.5,
        recall_min: float = 0.01,
        max_samples: float = 0.8,
        max_rules: int = 150,
        c: float = 0.01,
        min_support: float = 0.01,
        n_iter: int = 10000,
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.precision_min = precision_min
        self.recall_min = recall_min
        self.max_samples = max_samples
        self.max_rules = max_rules
        self.c = c
        self.min_support = min_support
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> OrdtClassifier:
        X = np.asarray(X, dtype=np.float64)

        # --- stage 1: mine candidate rules ---------------------------------
        t0 = time.perf_counter()
        miner = SkopeRules(
            feature_names=[f"x{i}" for i in range(X.shape[1])],
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            precision_min=self.precision_min,
            recall_min=self.recall_min,
            max_samples=self.max_samples,
            # keep near-duplicates: CORELS, not similarity heuristics,
            # decides which pool members earn a slot in the final list
            max_depth_duplication=None,
            random_state=self.random_state,
        )
        miner.fit(np.ascontiguousarray(X), y.astype(np.int64))
        raw_pool = miner.rules_without_feature_names_
        pretty_pool = miner.rules_
        mine_s = time.perf_counter() - t0
        n_raw = len(raw_pool)

        # --- stage 2: build capture matrix and select optimally -------------
        masks, f1s, labels, rules = [], [], [], []
        for (rule_str, scores), (pretty_str, _) in zip(raw_pool, pretty_pool):
            prec, rec = scores[0], scores[1]
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            mask = _rule_mask(rule_str, X)
            if not mask.any() or mask.all():
                continue  # degenerate columns carry no information
            masks.append(mask)
            f1s.append(f1)
            labels.append(pretty_str)
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
        corels.fit(Z, np.asarray(y).astype(np.uint8), features=self.pool_labels_)
        select_s = time.perf_counter() - t0

        self.corels_ = corels
        self.mine_s_, self.select_s_ = mine_s, select_s
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = np.stack([_rule_mask(r, X) for r in self.pool_rules_], axis=1).astype(
            np.uint8
        )
        return self.corels_.predict(Z)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def list_rules(self) -> list[tuple[str, bool]]:
        """Readable ordered rule list: (label, prediction); negated pool
        members are prefixed with NOT."""
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
        """(number of clauses, number of rules in the learned list)."""
        entries = [r for r in self.corels_.rl().rules if r["antecedents"]]
        return sum(len(r["antecedents"]) for r in entries), len(entries)


def compas_data() -> tuple[np.ndarray, np.ndarray]:
    path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "tests",
            "data",
            "compas.csv",
        )
    )
    X, y, _, _ = load_from_csv(path)
    return np.asarray(X), np.asarray(y)


def _rule_clauses(ruleset: list[tuple[str, Any]]) -> int:
    return sum(len(r.split(" and ")) for r, _ in ruleset)


def bench_dataset(
    name: str,
    kind: str,
    kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if kind == "breast-cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif kind == "compas":
        X, y = compas_data()
    else:
        X, y = make_classification(**kwargs)
    X, y = np.asarray(X), np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    rows: list[dict[str, Any]] = []

    def timed_fit(make_model, fit_args=(), fit_kwargs=None):
        best = None
        model = None
        for _ in range(args.repeats):
            model = make_model()
            t0 = time.perf_counter()
            model.fit(*fit_args, **(fit_kwargs or {}))
            elapsed = time.perf_counter() - t0
            best = elapsed if best is None else min(best, elapsed)
        return model, best

    # --- CART --------------------------------------------------------------
    cart, cart_s = timed_fit(
        lambda: DecisionTreeClassifier(random_state=0), (X_train, y_train)
    )
    rows.append(
        {
            "model": "cart",
            "fit_s": cart_s,
            "pred_ms": _pred_ms(cart, X_test),
            "test_acc": accuracy_score(y_test, cart.predict(X_test)),
            "train_acc": accuracy_score(y_train, cart.predict(X_train)),
            "size": f"{cart.get_n_leaves()} leaves",
            "clauses": None,
        }
    )

    # --- skope alone --------------------------------------------------------
    def make_skope():
        return SkopeRules(
            feature_names=[f"x{i}" for i in range(X.shape[1])],
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            precision_min=args.precision_min,
            recall_min=args.recall_min,
            max_depth_duplication=2,
            random_state=0,
        )

    skope, skope_s = timed_fit(make_skope, (X_train, y_train))
    skope_rules = skope.rules_
    rows.append(
        {
            "model": "skope",
            "fit_s": skope_s,
            "pred_ms": _pred_ms(skope, X_test),
            "test_acc": accuracy_score(y_test, skope.predict(X_test)),
            "train_acc": accuracy_score(y_train, skope.predict(X_train)),
            "size": f"{len(skope_rules)} rules",
            "clauses": _rule_clauses(skope_rules),
        }
    )

    # --- corels alone (own mining on one-hot bins) ---------------------------
    prebinarized = kwargs.get("prebinarized", False)
    if prebinarized:
        Xb_train, Xb_test = (
            X_train.astype(np.uint8),
            X_test.astype(np.uint8),
        )
        bin_s = None
    else:
        enc = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")
        t0 = time.perf_counter()
        Xb_train = (enc.fit_transform(X_train) > 0).astype(np.uint8)
        bin_s = time.perf_counter() - t0
        Xb_test = (enc.transform(X_test) > 0).astype(np.uint8)
    names = [f"x{i}" for i in range(Xb_train.shape[1])]

    def make_corels():
        return CorelsClassifier(
            c=args.c, min_support=args.min_support, max_card=1, verbosity=[]
        )

    corels, corels_s = timed_fit(
        make_corels, (Xb_train, y_train.astype(np.uint8)), {"features": names}
    )
    c_rules = [r for r in corels.rl().rules if r["antecedents"]]
    rows.append(
        {
            "model": "corels",
            "binarize_s": bin_s,
            "fit_s": corels_s,
            "pred_ms": _pred_ms(corels, Xb_test),
            "test_acc": accuracy_score(y_test, corels.predict(Xb_test)),
            "train_acc": accuracy_score(y_train, corels.predict(Xb_train)),
            "size": f"{len(c_rules)} rules",
            "clauses": sum(len(r["antecedents"]) for r in c_rules),
        }
    )

    # --- ORDT hybrid ---------------------------------------------------------
    def make_ordt():
        return OrdtClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            precision_min=args.precision_min,
            recall_min=args.recall_min,
            max_rules=args.max_rules,
            c=args.c,
            min_support=args.min_support,
            random_state=0,
        )

    ordt, _ = timed_fit(make_ordt, (X_train, y_train))
    clauses, n_list_rules = ordt.complexity()
    rows.append(
        {
            "model": "ordt",
            "mine_s": ordt.mine_s_,
            "select_s": ordt.select_s_,
            "fit_s": ordt.mine_s_ + ordt.select_s_,
            "pred_ms": _pred_ms(ordt, X_test),
            "test_acc": accuracy_score(y_test, ordt.predict(X_test)),
            "train_acc": accuracy_score(y_train, ordt.predict(X_train)),
            "size": f"{n_list_rules} rules",
            "clauses": clauses,
            "pool": f"{ordt.stats_['usable']} of {ordt.stats_['mined']}",
        }
    )
    _ = name
    return rows


def _pred_ms(model: Any, X: np.ndarray) -> float:
    model.predict(X)  # warm-up (imports, caches)
    t0 = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - t0) * 1e3


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'model':<7} {'mine s':>7} {'select s':>9} {'fit s':>7} {'pred ms':>8} "
        f"{'test acc':>9} {'train acc':>10} {'size':>12} {'clauses':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mine = f"{row['mine_s']:.3f}" if row.get("mine_s") is not None else "-"
        select = f"{row['select_s']:.3f}" if row.get("select_s") is not None else "-"
        clauses = "-" if row["clauses"] is None else str(row["clauses"])
        print(
            f"{row['model']:<7} {mine:>7} {select:>9} {row['fit_s']:>7.3f} "
            f"{row['pred_ms']:>8.2f} {row['test_acc']:>9.4f} "
            f"{row['train_acc']:>10.4f} {row['size']:>12} {clauses:>8}"
        )
    ordt_row = next((r for r in rows if r["model"] == "ordt"), None)
    if ordt_row and ordt_row.get("pool"):
        print(f"ordt candidate pool kept for CORELS: {ordt_row['pool']} rules")


WORKLOADS = [
    ("breast-cancer (real, n=569, d=30)", "breast-cancer", {}),
    (
        "small synthetic (n=2000, d=10)",
        "synthetic",
        {"n_samples": 2000, "n_features": 10, "n_informative": 6, "random_state": 1},
    ),
    (
        "medium synthetic (n=10000, d=20)",
        "synthetic",
        {"n_samples": 10000, "n_features": 20, "n_informative": 10, "random_state": 2},
    ),
    # compas features are already binary: CORELS skips its binarizer
    ("compas (real, binary, n=7214, d=27)", "compas", {"prebinarized": True}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--precision-min", type=float, default=0.5)
    parser.add_argument("--recall-min", type=float, default=0.01)
    parser.add_argument("--max-rules", type=int, default=150)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--min-support", type=float, default=0.01)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(
        f"ORDT benchmark (skope mining + CORELS selection; repeats={args.repeats}, "
        f"max_rules={args.max_rules}, c={args.c}, min_support={args.min_support}, "
        f"precision_min={args.precision_min})"
    )
    for name, kind, kwargs in WORKLOADS:
        print(flush=True)
        print(f"### {name}", flush=True)
        try:
            rows = bench_dataset(name, kind, dict(kwargs), args)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        print_table(rows)
        if kind == "compas":
            print_learned_list(name, kind, dict(kwargs), args)


def print_learned_list(name: str, kind: str, kwargs: dict, args) -> None:
    """Print the ORDT-learned rule list as an interpretability artifact."""
    if kind == "compas":
        X, y = compas_data()
    else:
        X, y = make_classification(**kwargs)
    X, y = np.asarray(X), np.asarray(y)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    ordt = OrdtClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        precision_min=args.precision_min,
        recall_min=args.recall_min,
        max_rules=args.max_rules,
        c=args.c,
        min_support=args.min_support,
        random_state=0,
    ).fit(X_train, y_train)
    print("\nORDT learned rule list (compas):")
    for label, pred in ordt.list_rules():
        print(f"  IF {label} THEN y={int(pred)}")
    print()


if __name__ == "__main__":
    main()
