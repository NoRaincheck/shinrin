#include "optimizer.hpp"

GosdtQueue::GosdtQueue(void) { return; }

GosdtQueue::~GosdtQueue(void) { return; }

// All operations run under the process-global shim mutex so that the
// membership filter and the priority queue are updated atomically with
// respect to other workers (and to graph sections holding accessors, which
// share the same lock).
bool GosdtQueue::push(Message const &message) {
    std::lock_guard<std::recursive_mutex> guard(tbb::shim_lock());
    message_type *internal_message = new message_type();
    *internal_message = message;
    // Attempt to copy content into membership set
    if (this->membership.insert(std::make_pair(internal_message, true))) {
        this->queue.push(internal_message);
        return true;
    } else {
        delete internal_message;
        return false;
    }
}

bool GosdtQueue::empty(void) const {
    std::lock_guard<std::recursive_mutex> guard(tbb::shim_lock());
    return this->queue.empty();
}

unsigned int GosdtQueue::size(void) const {
    std::lock_guard<std::recursive_mutex> guard(tbb::shim_lock());
    return this->queue.size();
}

bool GosdtQueue::pop(Message &message) {
    std::lock_guard<std::recursive_mutex> guard(tbb::shim_lock());
    message_type *internal_message;
    if (this->queue.try_pop(internal_message)) {
        this->membership.erase(internal_message);  // remove membership
        message = *internal_message;

        delete internal_message;
        return true;
    } else {
        return false;
    }
}
