/*
 * Shim for oneTBB's concurrent_vector. The GOSDT engine only uses the
 * sequential subset (push_back, iteration, size) and only ever touches these
 * containers as mapped VALUES of a tbb::concurrent_hash_map (bound lists,
 * i.e. reachable exclusively through that map's accessors). They are thereby
 * covered by the shim's global mutex and need no internal locking.
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_VECTOR_H
#define SHINRIN_TBB_SHIM_CONCURRENT_VECTOR_H

#include <vector>

namespace tbb {

template <typename T, typename Allocator = std::allocator<T>>
class concurrent_vector : public std::vector<T, Allocator> {
   public:
    using std::vector<T, Allocator>::vector;

    void grow_by(typename std::vector<T, Allocator>::size_type delta) {
        this->resize(this->size() + delta);
    }
};

}  // namespace tbb

#endif
