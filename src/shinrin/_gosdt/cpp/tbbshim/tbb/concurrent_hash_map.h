/*
 * Serial shim for oneTBB's concurrent_hash_map. Provides the subset of the
 * TBB interface used by the vendored GOSDT engine:
 *
 *   - insert(accessor&, key) / find(accessor&, key) with accessor objects
 *     exposing ->first / ->second
 *   - insert(value_type const&) returning true when a new entry was created
 *   - erase(key), clear(), size(), empty()
 *
 * Hashing/equality delegate to the TBB-style comparator's static hash() and
 * equal() members. The engine is run single-threaded (worker_limit = 1), so
 * no locking is required.
 */
#ifndef SHINRIN_TBB_SHIM_CONCURRENT_HASH_MAP_H
#define SHINRIN_TBB_SHIM_CONCURRENT_HASH_MAP_H

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

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
    class accessor {
       public:
        accessor() = default;

        value_type *operator->() const { return entry_.operator->(); }
        value_type &operator*() const { return *entry_; }
        // Serial equivalent of releasing the TBB lock: invalidate.
        void release() { entry_ = entry<typename map_type::iterator>(); }

       private:
        friend class concurrent_hash_map;
        explicit accessor(typename map_type::iterator it) : entry_(it) {}
        entry<typename map_type::iterator> entry_;
    };

    class const_accessor {
       public:
        const_accessor() = default;

        value_type const *operator->() const { return entry_.operator->(); }
        value_type const &operator*() const { return *entry_; }
        void release() { entry_ = entry<typename map_type::const_iterator>(); }

       private:
        friend class concurrent_hash_map;
        explicit const_accessor(typename map_type::const_iterator it) : entry_(it) {}
        entry<typename map_type::const_iterator> entry_;
    };

    concurrent_hash_map() = default;

    // Inserts a default-constructed value for key if absent and returns true;
    // otherwise binds the accessor to the existing entry and returns false.
    bool insert(accessor &result, key_type const &key) {
        auto [it, inserted] = map_.try_emplace(key);
        result = accessor(it);
        return inserted;
    }

    bool find(const_accessor &result, key_type const &key) const {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        result = const_accessor(it);
        return true;
    }

    bool find(accessor &result, key_type const &key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        result = accessor(it);
        return true;
    }

    // Inserts the given value; binds the accessor to the entry (existing or
    // newly created) and returns true only when a new entry was inserted.
    bool insert(accessor &result, value_type const &value) {
        auto [it, inserted] = map_.try_emplace(value.first, value.second);
        result = accessor(it);
        return inserted;
    }

    bool insert(const_accessor &result, value_type const &value) {
        auto [it, inserted] = map_.try_emplace(value.first, value.second);
        result = const_accessor(it);
        return inserted;
    }

    // Inserts the given value; returns true only if the key was not present.
    bool insert(value_type const &value) { return map_.insert(value).second; }

    bool erase(key_type const &key) { return map_.erase(key) > 0; }

    void clear() { map_.clear(); }
    size_t size() const { return map_.size(); }
    bool empty() const { return map_.empty(); }

   private:
    map_type map_;
};

}  // namespace tbb

#endif
