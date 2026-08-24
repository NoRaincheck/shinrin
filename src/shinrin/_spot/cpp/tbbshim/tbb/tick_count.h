/*
 * Serial shim for oneTBB's tick_count timing utility. The vendored GOSDT
 * engine includes the header but performs its timing with std::chrono; this
 * stub keeps the include path valid.
 */
#ifndef SHINRIN_TBB_SHIM_TICK_COUNT_H
#define SHINRIN_TBB_SHIM_TICK_COUNT_H

#include <chrono>

namespace tbb {

class tick_count {
   public:
    class interval_t {
       public:
        double seconds() const { return seconds_; }

       private:
        friend class tick_count;
        explicit interval_t(double seconds) : seconds_(seconds) {}
        double seconds_;
    };

    static tick_count now() {
        return tick_count(
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    tick_count() : seconds_(0.0) {}

    interval_t operator-(tick_count const &other) const { return interval_t(seconds_ - other.seconds_); }

   private:
    explicit tick_count(double seconds) : seconds_(seconds) {}
    double seconds_;
};

}  // namespace tbb

#endif
