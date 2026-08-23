/*
 * Serial shim for oneTBB's concurrent_unordered_map. The GOSDT engine only
 * uses the std-style sequential subset (insert, find, iteration); with
 * worker_limit = 1 no concurrent access occurs.
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_UNORDERED_MAP_H
#define SHINRIN_TBB_SHIM_CONCURRENT_UNORDERED_MAP_H

#include <functional>
#include <memory>
#include <unordered_map>

namespace tbb {

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Equal = std::equal_to<Key>,
          typename Allocator = std::allocator<std::pair<Key const, Value>>>
using concurrent_unordered_map = std::unordered_map<Key, Value, Hash, Equal, Allocator>;

}  // namespace tbb

#endif
