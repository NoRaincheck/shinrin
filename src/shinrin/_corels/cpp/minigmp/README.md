# Vendored mini-gmp

[mini-gmp](https://gmplib.org/manual/mini_002dgmp.html) is GMP's own small,
portable implementation of a subset of the GMP API (`mpz_t` bignums) in two
files (`mini-gmp.{c,h}`). It is vendored here from the official GMP **6.3.0**
release tarball (<https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz>,
`mini-gmp/mini-gmp.c`, `mini-gmp/mini-gmp.h`) unmodified.

This allows CORELS to be compiled with `-DGMP` (the faster bit-vector /
search-space-size code path) while keeping the built extension fully
self-contained: there is **no dependency on a system libgmp** at build or run
time, so PyPI wheels need no intermediary packages.

Note: mini-gmp trades some of GMP's hand-tuned assembly speed for portability;
it is slower than a system libgmp for very large operands, which only affects
the search-space size estimates and very wide bit vectors.

License: LGPL-3.0-or-later (see `LICENSE`, reproduced from the GMP tarball).
