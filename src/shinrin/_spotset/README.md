# Vendored SPOTSET engine (treeFARMS)

This directory vendors [ubc-systopia/treeFARMS](https://github.com/ubc-systopia/treeFARMS)
(BSD-3-Clause, see `LICENSE`) — *Exploring the Whole Rashomon Set of Sparse
Decision Trees* (McTavish et al., NeurIPS 2022) — renamed **SPOTSET**
(**Sparse Optimal Rashomon Trees**).

> Naming note: treeFARMS ("Trees FAst RashoMon Sets") was itself built on
> [gosdt-guesses](https://github.com/ubc-systopia/gosdt-guesses), and upstream
> names nearly everything "GOSDT" (`libgosdt`, `class GOSDT`, …). To avoid
> confusion with this project's single-optimal-tree trainer
> [`SPOTClassifier`](../_spot/) (formerly `GOSDTClassifier`), the vendored
> engine is renamed **SPOTSET**: it enumerates the *set* of near-optimal trees
> (the Rashomon set) instead of stopping at a single optimum.

## Layout

- `engine/` – the C++ engine (upstream `src/`), vendored almost as-is.
  Upstream's `main.cpp`/`main.hpp` (CLI driver) and
  `python_extension.cpp`/`python_extension.hpp` (CPython binding) are not
  vendored; they are replaced by the bridge + PyO3 layer described below.
- `csv/csv.h` – Ben Strasser's single-header CSV parser (BSD-3), required by
  `dataset.hpp`/`encoder.hpp`.
- `LICENSE` – upstream BSD-3-Clause license text.

## Integration notes / deviations

- **Namespace**: because SPOTSET shares its gosdt-guesses lineage with the
  already-vendored SPOT engine (`src/shinrin/_spot/cpp/libspot`), every engine
  file is wrapped in `namespace spotset`; without it the two engines' identically
  named global classes (`Bitmask`, `Task`, `Tile`, `Graph`, `Queue`, …) would
  collide at link time. The `namespace std` hash/equality specializations for
  engine types stay at global scope (required by the language) but reference
  the types as `spotset::…`.
- **Bridge/bindings**: upstream's pybind11 module (`libgosdt`) is replaced by a
  C ABI bridge plus PyO3 bindings in `src/lib.rs` (`shinrin._native.spotset_fit`),
  mirroring how the SPOT engine is integrated. No Python.h dependency remains.
- **Shared dependencies**: the engine reuses this repo's lock-based TBB shim
  (`src/shinrin/_spot/cpp/tbbshim`), bundled mini-gmp + mpn logical ops
  (`SHINRIN_CORELS_NO_GMP` does not affect this engine either), and nlohmann/json.
  Upstream bundles json 3.7.0; we compile against the shared nlohmann 3.11.3
  copy to keep a single version per binary (the `<json/json.hpp>` includes were
  rewritten to `<nlohmann/json.hpp>`).
- **Vestigial include dropped**: `graph.hpp` includes
  `<tbb/concurrent_unordered_set.h>` but never uses the type; the include was
  removed instead of adding an unused shim.
- **tbb::tick_count**: gained a default constructor in the shared shim
  (oneTBB's type is default-constructible too); treeFARMS' `Optimizer` stores a
  `start_time` member that relies on it.
- The imbalance/OSDT pure-Python variant (`treefarms/model/imbalance/*`,
  ~2.9k lines) and the timbertrek visualization dependency are intentionally
  not ported; see the Python layer notes in `src/shinrin/_spotset`.

Upstream version: main at vendoring time (196 commits).
