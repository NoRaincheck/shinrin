/*
 * Shared process-global lock for the tbb shim containers. Extracted into
 * its own header so each container shim is self-contained regardless of
 * the order in which engine headers include them.
 */
#ifndef SHINRIN_TBB_SHIM_SHIM_LOCK_H
#define SHINRIN_TBB_SHIM_SHIM_LOCK_H

#include <mutex>

namespace tbb {

// Single process-wide lock shared by all shim containers. An inline function
// with a static local guarantees exactly one instance across translation
// units, and magic statics make initialization thread-safe.
inline std::recursive_mutex &shim_lock(void) {
    static std::recursive_mutex lock;
    return lock;
}

}  // namespace tbb

#endif
