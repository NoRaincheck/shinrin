/*
 * Limb-wise logical operations required by the vendored GOSDT engine that
 * mini-gmp does not expose. Semantics follow GMP's documented mpn functions;
 * destination pointers may alias the sources.
 *
 * Part of the shinrin GMP shim (no system libgmp dependency).
 */
#include <string.h>

#include "gmp.h"

#define SHINRIN_MPN_LOGICAL(OP)                                  \
    do {                                                         \
        for (mp_size_t i = 0; i < n; i++) {                      \
            rp[i] = ~(s1p[i] OP s2p[i]);                         \
        }                                                        \
    } while (0)

void mpn_and_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    for (mp_size_t i = 0; i < n; i++) {
        rp[i] = s1p[i] & s2p[i];
    }
}

void mpn_ior_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    for (mp_size_t i = 0; i < n; i++) {
        rp[i] = s1p[i] | s2p[i];
    }
}

void mpn_xor_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    for (mp_size_t i = 0; i < n; i++) {
        rp[i] = s1p[i] ^ s2p[i];
    }
}

void mpn_xnor_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    SHINRIN_MPN_LOGICAL(^);
}

void mpn_nand_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    SHINRIN_MPN_LOGICAL(&);
}

void mpn_nior_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n) {
    SHINRIN_MPN_LOGICAL(|);
}
