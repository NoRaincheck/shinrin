#!/usr/bin/env python3
"""Attribute Mojo TabICL inference time with macOS ``sample``.

Runs a fixed ``TabICLClassifier`` workload on the ``mojo`` backend in a
child process, samples its native stacks with ``sample`` (no root
required) while it loops predictions, then parses the "Sort by top of
stack" section — each entry is the number of samples whose top frame is
that symbol, i.e. self-time attribution across all threads.

Usage:

    python scripts/benchmarks/profile_tabicl_mojo.py [--duration S]
        [--train N] [--features N] [--test N] [--n-estimators N]
        [--batch-size N] [--raw PATH]

Example output row: share of all samples spent at the top of the stack
inside ``gemm_nt_rows`` (in ``_native_tabicl.so``).
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def child_workload(args: argparse.Namespace) -> None:
    """Fit once, then loop predictions until terminated."""
    import numpy as np

    from shinrin.tabicl import TabICLClassifier

    rng = np.random.RandomState(0)
    X = rng.randn(args.train + args.test, args.features).astype(np.float32)
    logits = X @ rng.randn(args.features).astype(np.float32)
    y = (logits > np.median(logits)).astype(np.int64)

    clf = TabICLClassifier(
        backend="mojo",
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        random_state=42,
    ).fit(X[: args.train], y[: args.train])
    X_test = X[args.train :]
    while True:
        clf.predict_proba(X_test)


_TOP_OF_STACK = re.compile(r"^\s+(.+?)\s+\(in ([^)]+)\)\s+(\d+)\s*$")

WAIT_SYMBOLS = frozenset(
    {
        "semaphore_wait_trap",
        "__workq_kernreturn",
        "__ulock_wait",
        "__psynch_cvwait",
        "thread_start",
        "_thread_start",
        "start_wqthread",
    }
)


def short_name(sym: str) -> str:
    """Strip demangled signatures / module prefixes for display."""
    sym = sym.split("(", 1)[0]
    sym = sym.replace("shinrin::", "")
    while sym.startswith(":"):
        sym = sym[1:]
    return sym or "<unnamed>"


def parse_top_of_stack(raw: str) -> dict[str, int]:
    """Parse the 'Sort by top of stack' section into {symbol: samples}."""
    marker = "Sort by top of stack"
    start = raw.find(marker)
    if start < 0:
        raise RuntimeError("sample output missing 'Sort by top of stack'")
    section = raw[start : raw.find("\n\n", start)]
    counts: dict[str, int] = {}
    for line in section.splitlines():
        m = _TOP_OF_STACK.match(line)
        if m:
            sym = m.group(1)
            counts[sym] = counts.get(sym, 0) + int(m.group(3))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        child_workload(args)
        return

    child = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--child",
            *[
                str(a)
                for a in (
                    "--train",
                    args.train,
                    "--features",
                    args.features,
                    "--test",
                    args.test,
                    "--n-estimators",
                    args.n_estimators,
                    "--batch-size",
                    args.batch_size,
                )
            ],
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3.0)  # allow imports + checkpoint load + first predict
        raw_path = args.raw or Path("/tmp/tabicl_sample.txt")
        subprocess.run(
            ["sample", str(child.pid), str(args.duration), "-file", str(raw_path)],
            check=True,
            capture_output=True,
        )
    finally:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)

    counts = parse_top_of_stack(raw_path.read_text(errors="replace"))
    total = sum(counts.values())
    print(
        f"samples: {total} over {args.duration}s "
        f"(workload {args.train}x{args.features}, test {args.test})"
    )

    print("\n== all thread-time (incl. scheduler parks) ==")
    print(f"{'samples':>8} {'share':>7}  symbol")
    for sym, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{n:>8} {100.0 * n / max(total, 1):>6.1f}%  {short_name(sym)}")

    compute = {s: n for s, n in counts.items() if s not in WAIT_SYMBOLS}
    ctotal = sum(compute.values())
    print(
        f"\n== compute only ({ctotal} samples, "
        f"{100.0 * ctotal / max(total, 1):.1f}% of thread-time) =="
    )
    print(f"{'samples':>8} {'share':>7}  symbol")
    for sym, n in sorted(compute.items(), key=lambda kv: -kv[1])[:25]:
        print(f"{n:>8} {100.0 * n / max(ctotal, 1):>6.1f}%  {short_name(sym)}")


if __name__ == "__main__":
    main()
