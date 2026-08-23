/*
 * Serial shim for oneTBB's concurrent_priority_queue. Matches the TBB
 * semantics used by the vendored GOSDT engine: push() enqueues and try_pop()
 * removes the element with the highest priority (i.e. the element for which
 * the comparator orders "greatest").
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_PRIORITY_QUEUE_H
#define SHINRIN_TBB_SHIM_CONCURRENT_PRIORITY_QUEUE_H

#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

namespace tbb {

template <typename T, typename Compare = std::less<T>,
          typename Allocator = std::allocator<T>>
class concurrent_priority_queue {
   public:
    concurrent_priority_queue() = default;

    void push(T const &value) { queue_.push(value); }

    bool try_pop(T &out) {
        if (queue_.empty()) {
            return false;
        }
        out = queue_.top();
        queue_.pop();
        return true;
    }

    bool empty() const { return queue_.empty(); }
    size_t size() const { return queue_.size(); }

   private:
    std::priority_queue<T, std::vector<T, Allocator>, Compare> queue_;
};

}  // namespace tbb

#endif
