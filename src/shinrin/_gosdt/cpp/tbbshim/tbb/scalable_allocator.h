/*
 * Shim for oneTBB's scalable_allocator. Maps to std::allocator; no
 * concurrent allocator is required (GMP-backed bitmasks allocate through
 * libc malloc, which is thread-safe).
 */
#ifndef SHINRIN_TBB_SHIM_SCALABLE_ALLOCATOR_H
#define SHINRIN_TBB_SHIM_SCALABLE_ALLOCATOR_H

#include <memory>

namespace tbb {

template <typename T>
using scalable_allocator = std::allocator<T>;

}  // namespace tbb

#endif
