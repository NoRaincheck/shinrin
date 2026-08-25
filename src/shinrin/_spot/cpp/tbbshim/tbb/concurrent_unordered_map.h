/*
 * Shim for oneTBB's concurrent_unordered_map. The GOSDT engine only uses the
 * std-style sequential subset (insert, find, iteration) and only ever touches
 * these containers as mapped VALUES of a tbb::concurrent_hash_map (adjacency
 * sets, i.e. reachable exclusively through that map's accessors). They are
 * thereby covered by the shim's global mutex and need no internal locking.
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
