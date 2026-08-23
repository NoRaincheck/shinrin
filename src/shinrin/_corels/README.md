# Vendored pycorels (CORELS)

This directory vendors [pycorels](https://github.com/corels/pycorels), the
Python binding of the CORELS (Certifiably Optimal RulE ListS) algorithm,
so that shinrin can offer optimal rule lists without an external dependency.

Upstream version: pycorels master (GPL-3.0, see `LICENSE`).

## Layout

- `cpp/corels/` – the CORELS C++ engine vendored from
  `pycorels/corels/src/corels/src` (the CLI entrypoint `main.cpp`, the
  GNUmakefiles and the upstream C++ test suite are not used).
- `cpp/mining/` – rule mining / minority bound helpers vendored from
  `pycorels/corels/src/utils.{hh,cpp}`.
- `cpp/minigmp/` – [mini-gmp](cpp/minigmp/README.md), GMP's portable `mpz_t`
  implementation, vendored unmodified from the official GMP 6.3.0 tarball.
- `cpp/gmpshim/gmp.h` – shim redirecting CORELS' `#include <gmp.h>` to
  mini-gmp.
- `cpp/bridge.cpp` – new C ABI bridge replacing upstream's Cython module
  (`corels/_corels.pyx`). It is compiled into the `shinrin._native`
  extension by `build.rs` together with the C++ sources.
- `corels.py`, `utils.py` – adapted copies of the upstream Python layer.
- `LICENSE` – the upstream GPL-3.0 license text.

## Bundled GMP via mini-gmp

Upstream optionally uses GMP (`mpz_t`) to represent rule bit vectors when
compiled with `-DGMP`, which speeds up bit-vector operations and enables the
search-space size estimates. This vendoring compiles with `-DGMP` against
vendored mini-gmp, so the GMP code path is active while the built extension
stays fully self-contained: **no system libgmp** is needed at build or run
time, and PyPI wheels require no intermediary packages.

## Deviations from upstream

- The Cython extension is replaced by PyO3 bindings (`src/lib.rs`) with
  identical semantics; the stray debug `print(rulelist.size())` in
  `fit_wrap_end` is dropped.
- Python code updated for NumPy >= 2 (`np.bool` removed, `copy=False`
  semantics) and modernized lint compliance.
- `predict` returns a plain `bool` dtype array (upstream used the removed
  `np.bool` alias).
