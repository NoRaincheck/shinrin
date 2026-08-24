/*
 * Thread-safe shim for oneTBB's concurrent_priority_queue. Matches the TBB
 * semantics used by the vendored GOSDT engine: push() enqueues and try_pop()
 * removes the element with the highest priority (i.e. the element for which
 * the comparator orders "greatest").
 *
 * All operations are serialized with the process-global shim mutex shared
 * with tbb::concurrent_hash_map (see concurrent_hash_map.h), which keeps the
 * lock hierarchy trivial: engine code may push onto the queue while holding
 * graph accessors without risk of deadlock.
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_PRIORITY_QUEUE_H
#define SHINRIN_TBB_SHIM_CONCURRENT_PRIORITY_QUEUE_H

#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

namespace tbb {

template <typename T, typename Compare = std::less<T>,
          typename Allocator = std::allocator<T>>
class concurrent_priority_queue {
   public:
    concurrent_priority_queue() = default;

    void push(T const &value) {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        queue_.push(value);
    }

    bool try_pop(T &out) {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        if (queue_.empty()) {
            return false;
        }
        out = queue_.top();
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return queue_.empty();
    }
    size_t size() const {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return queue_.size();
    }

   private:
    std::priority_queue<T, std::vector<T, Allocator>, Compare> queue_;
};

}  // namespace tbb

#endif
