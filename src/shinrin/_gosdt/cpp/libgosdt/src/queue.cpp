#include "optimizer.hpp"

GosdtQueue::GosdtQueue(void) { return; }

GosdtQueue::~GosdtQueue(void) { return; }

bool GosdtQueue::push(Message const &message) {
    message_type *internal_message = new message_type();
    *internal_message = message;
    std::lock_guard<std::mutex> guard(q_mutex);
    // Attempt to copy content into membership set
    if (this->membership.insert(std::make_pair(internal_message, true))) {
        this->queue.push(internal_message);
        return true;
    } else {
        delete internal_message;
        return false;
    }
}

bool GosdtQueue::empty(void) const { return size() == 0; }

unsigned int GosdtQueue::size(void) const { return this->queue.size(); }

bool GosdtQueue::pop(Message &message) {
    std::lock_guard<std::mutex> guard(q_mutex);
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
