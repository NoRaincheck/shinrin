/*
 * Serial shim for oneTBB's scalable_allocator, used by the vendored GOSDT
 * engine. Maps to std::allocator; the engine is run single-threaded
 * (worker_limit = 1) so no concurrent allocator is required.
 */
#ifndef SHINRIN_TBB_SHIM_SCALABLE_ALLOCATOR_H
#define SHINRIN_TBB_SHIM_SCALABLE_ALLOCATOR_H

#include <memory>

namespace tbb {

template <typename T>
using scalable_allocator = std::allocator<T>;

}  // namespace tbb

#endif
