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
- `cpp/bridge.cpp` – new C ABI bridge replacing upstream's Cython module
  (`corels/_corels.pyx`). It is compiled into the `shinrin._native`
  extension by `build.rs` together with the C++ sources.
- `corels.py`, `utils.py` – adapted copies of the upstream Python layer.
- `LICENSE` – the upstream GPL-3.0 license text.

## No GMP

Upstream optionally uses GMP (`mpz_t`) to represent rule bit vectors when
compiled with `-DGMP`. This vendoring never defines `GMP`, so CORELS uses its
built-in word-array (`v_entry*`) bit vector fallback. There is **no libgmp
dependency** at build or run time; upstream documents this configuration as
supported but slower for very large search spaces.

## Deviations from upstream

- The Cython extension is replaced by PyO3 bindings (`src/lib.rs`) with
  identical semantics; the stray debug `print(rulelist.size())` in
  `fit_wrap_end` is dropped.
- Python code updated for NumPy >= 2 (`np.bool` removed, `copy=False`
  semantics) and modernized lint compliance.
- `predict` returns a plain `bool` dtype array (upstream used the removed
  `np.bool` alias).
