/*
 * Shim so that CORELS' `#include <gmp.h>` (active when compiled with
 * -DGMP) resolves to the vendored mini-gmp instead of requiring libgmp
 * to be installed on the system.
 */
#pragma once

#include "../minigmp/mini-gmp.h"
