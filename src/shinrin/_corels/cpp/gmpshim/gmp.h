/*
 * Shim so that CORELS' `#include <gmp.h>` (active when compiled with
 * -DGMP) resolves to the vendored mini-gmp instead of requiring libgmp
 * to be installed on the system.
 */
#pragma once

#include "../minigmp/mini-gmp.h"

/* ---------------------------------------------------------------------------
 * Minimal additions used by the vendored GOSDT engine (see
 * src/shinrin/_gosdt): the six limb-wise logical mpn operations. mini-gmp
 * declares scan/copy/popcount/cmp but not these, so they are provided by
 * gmpshim/mpn_logical.c compiled into the extension.
 */
#ifdef __cplusplus
extern "C" {
#endif

void mpn_and_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_ior_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_xor_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_xnor_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_nand_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_nior_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#ifdef __cplusplus
}
#endif
