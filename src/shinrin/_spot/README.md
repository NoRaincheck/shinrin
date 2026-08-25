# Vendored gosdt-guesses (GOSDT)

This directory vendors [ubc-systopia/gosdt-guesses](https://github.com/ubc-systopia/gosdt-guesses),
the canonical implementation of:

- McTavish et al., *Fast Sparse Decision Tree Optimization via Reference
  Ensembles*, AAAI 2022 — reference-ensemble guesses (e.g. from a boosted
  ensemble) bound the optimal-tree search,
- Lin et al., *Generalized and Scalable Optimal Sparse Decision Trees*, ICML
  2020 — the underlying GOSDT branch-and-bound.

Upstream version: master at vendoring time (BSD-3-Clause, see `LICENSE`).

## Layout

- `cpp/libgosdt/` – the C++ engine (headers + sources), vendored as-is.
- `cpp/nlohmann/` – [nlohmann/json](https://github.com/nlohmann/json)
  v3.11.3 single header (MIT, see `nlohmann/LICENSE.MIT`).
- `cpp/tbbshim/tbb/` – lock-based replacements for the oneTBB container types
  the engine uses (`concurrent_hash_map`, `concurrent_unordered_map`,
  `concurrent_vector`, `concurrent_priority_queue`, `scalable_allocator`,
  `tick_count`). Accessors retain a process-global recursive mutex from
  acquisition to `release()`, reproducing TBB's exclusive-entry semantics
  with coarse granularity; no TBB dependency remains.
- `classifier.py`, `threshold_guessing.py`, `binarizer.py`, `tree.py`,
  `status.py` – adapted copies of the upstream Python layer.
- `LICENSE` – upstream BSD-3-Clause license text.

## Integration notes / deviations

- Upstream's pybind11 module (`_libgosdt`) is replaced by a C ABI bridge
  (`cpp/bridge_gosdt.cpp`) plus PyO3 bindings in `src/lib.rs`
  (`shinrin._native.gosdt_fit`).
- The engine honours `worker_limit`: 1 (default) runs the search
  single-threaded, values above 1 spin up parallel branch-and-bound workers,
  and 0 uses one worker per available core. See
  `scripts/benchmarks/GOSDT_BENCHMARK.md` for measured scaling.
- GOSDT's global `class Queue` is renamed to `GosdtQueue` in the vendored
  copy: CORELS (also vendored) defines an unrelated global `Queue`, and the
  identical C++ symbol names caused the linker to silently mix the two
  implementations, corrupting memory.
- GMP: the engine unconditionally includes `<gmp.h>` and uses low-level
  `mpn_*` limb operations. These resolve to the bundled mini-gmp plus six
  logical ops provided by `src/shinrin/_corels/cpp/gmpshim/mpn_logical.c`.
  No system libgmp is required; the `SHINRIN_CORELS_NO_GMP` build toggle does
  not affect this engine.
- The sklearn compatibility surface (`check_estimator` for the binarizers)
  and hyperparameters are preserved; `debug=True` saves CSV dumps of the raw
  inputs but not the binary dataset/config snapshots (those helpers were part
  of the removed pybind layer).
