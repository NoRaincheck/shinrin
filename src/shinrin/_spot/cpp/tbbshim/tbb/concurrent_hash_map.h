/*
 * Thread-safe shim for oneTBB's concurrent_hash_map. Provides the subset of
 * the TBB interface used by the vendored GOSDT engine:
 *
 *   - insert(accessor&, key) / find(accessor&, key) with accessor objects
 *     exposing ->first / ->second
 *   - insert(value_type const&) returning true when a new entry was created
 *   - erase(key), clear(), size(), empty()
 *
 * Hashing/equality delegate to the TBB-style comparator's static hash() and
 * equal() members.
 *
 * Synchronisation: TBB grants accessors exclusive ownership of their entry
 * between acquisition and release(). Reproducing per-entry locks would mean
 * reimplementing TBB, so instead every operation takes a process-global
 * recursive mutex and accessors HOLD that mutex from acquisition until
 * release() (or destruction). This preserves the engine's nesting patterns
 * (e.g. vertices -> bounds -> children) without deadlocks: there is a single
 * lock, so no lock-order cycles can form. All heavy computation in the engine
 * happens outside accessor lifetimes, so contention stays low in practice.
 *
 * Note: instances of tbb::concurrent_vector and tbb::concurrent_unordered_map
 * are only ever used as mapped VALUES of these hash maps (adjacency sets,
 * bound lists) and are therefore covered by the same mutex whenever they are
 * reachable through an accessor.
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_HASH_MAP_H
#define SHINRIN_TBB_SHIM_CONCURRENT_HASH_MAP_H

#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shim_lock.h"

namespace tbb {

template <typename Key, typename Value, typename HashCompare,
          typename Allocator = std::allocator<std::pair<Key const, Value>>>
class concurrent_hash_map {
   public:
    typedef Key key_type;
    typedef std::pair<Key const, Value> value_type;

   private:
    struct Hasher {
        size_t operator()(key_type const &key) const { return HashCompare::hash(key); }
    };
    struct Equaler {
        bool operator()(key_type const &left, key_type const &right) const {
            return HashCompare::equal(left, right);
        }
    };
    typedef std::unordered_map<key_type, Value, Hasher, Equaler,
                               typename std::allocator_traits<Allocator>::template rebind_alloc<
                                   std::pair<key_type const, Value>>>
        map_type;

    // Iterator proxy modelling the validity state of a TBB accessor.
    template <typename Iterator>
    class entry {
       public:
        entry() : valid_(false) {}
        explicit entry(Iterator it) : iterator_(it), valid_(true) {}

        value_type *operator->() const { return &*iterator_; }
        value_type &operator*() const { return *iterator_; }
        operator bool() const { return valid_; }

       private:
        Iterator iterator_;
        bool valid_;
    };

   public:
    // Accessors are movable-only handles that retain the shim lock from
    // acquisition until release() or destruction, matching TBB's guarantee
    // that the referenced entry cannot be modified concurrently.
    class accessor {
       public:
        accessor() = default;
        accessor(accessor &&other) = default;
        accessor &operator=(accessor &&other) = default;
        accessor(accessor const &) = delete;
        accessor &operator=(accessor const &) = delete;

        value_type *operator->() const { return entry_.operator->(); }
        value_type &operator*() const { return *entry_; }
        // Releases the retained lock and invalidates the accessor. A no-op
        // when the accessor was never bound (the engine calls release()
        // unconditionally after failed lookups as well).
        void release() {
            if (lock_.owns_lock()) {
                lock_.unlock();
            }
            entry_ = entry<typename map_type::iterator>();
        }

       private:
        friend class concurrent_hash_map;
        accessor(typename map_type::iterator it, std::unique_lock<std::recursive_mutex> &&lock)
            : entry_(it), lock_(std::move(lock)) {}
        entry<typename map_type::iterator> entry_;
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class const_accessor {
       public:
        const_accessor() = default;
        const_accessor(const_accessor &&other) = default;
        const_accessor &operator=(const_accessor &&other) = default;
        const_accessor(const_accessor const &) = delete;
        const_accessor &operator=(const_accessor const &) = delete;

        value_type const *operator->() const { return entry_.operator->(); }
        value_type const &operator*() const { return *entry_; }
        void release() {
            if (lock_.owns_lock()) {
                lock_.unlock();
            }
            entry_ = entry<typename map_type::const_iterator>();
        }

       private:
        friend class concurrent_hash_map;
        const_accessor(typename map_type::const_iterator it, std::unique_lock<std::recursive_mutex> &&lock)
            : entry_(it), lock_(std::move(lock)) {}
        entry<typename map_type::const_iterator> entry_;
        std::unique_lock<std::recursive_mutex> lock_;
    };

    concurrent_hash_map() = default;

    // Inserts a default-constructed value for key if absent and returns true;
    // otherwise binds the accessor to the existing entry and returns false.
    // In both cases the accessor retains the lock until release().
    bool insert(accessor &result, key_type const &key) {
        std::unique_lock<std::recursive_mutex> lock(shim_lock());
        auto [it, inserted] = map_.try_emplace(key);
        result = accessor(it, std::move(lock));
        return inserted;
    }

    bool find(const_accessor &result, key_type const &key) const {
        std::unique_lock<std::recursive_mutex> lock(shim_lock());
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        result = const_accessor(it, std::move(lock));
        return true;
    }

    bool find(accessor &result, key_type const &key) {
        std::unique_lock<std::recursive_mutex> lock(shim_lock());
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        result = accessor(it, std::move(lock));
        return true;
    }

    // Inserts the given value; binds the accessor to the entry (existing or
    // newly created) and returns true only when a new entry was inserted.
    bool insert(accessor &result, value_type const &value) {
        std::unique_lock<std::recursive_mutex> lock(shim_lock());
        auto [it, inserted] = map_.try_emplace(value.first, value.second);
        result = accessor(it, std::move(lock));
        return inserted;
    }

    bool insert(const_accessor &result, value_type const &value) {
        std::unique_lock<std::recursive_mutex> lock(shim_lock());
        auto [it, inserted] = map_.try_emplace(value.first, value.second);
        result = const_accessor(it, std::move(lock));
        return inserted;
    }

    // Inserts the given value; returns true only if the key was not present.
    bool insert(value_type const &value) {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return map_.insert(value).second;
    }

    bool erase(key_type const &key) {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return map_.erase(key) > 0;
    }

    void clear() {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        map_.clear();
    }
    size_t size() const {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return map_.size();
    }
    bool empty() const {
        std::lock_guard<std::recursive_mutex> guard(shim_lock());
        return map_.empty();
    }

   private:
    map_type map_;
};

}  // namespace tbb

#endif
