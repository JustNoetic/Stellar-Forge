#pragma once

#include <cstdint>
#include <deque>
#include <optional>

namespace sf::rhi {

class UploadRing {
public:
    explicit UploadRing(uint64_t capacity = 0) : capacity_(capacity) {}
    std::optional<uint64_t> allocate(uint64_t bytes, uint64_t serial, uint64_t completed) {
        while (!live_.empty() && live_.front().serial <= completed) {
            tail_ = live_.front().end;
            live_.pop_front();
        }
        if (live_.empty()) tail_ = head_;
        if (bytes == 0 || bytes > capacity_ || serial <= completed) return {};
        uint64_t begin = (head_ + 15) & ~uint64_t(15);
        if (begin % capacity_ + bytes > capacity_) begin += capacity_ - begin % capacity_;
        if (begin + bytes - tail_ > capacity_) return {};
        head_ = begin + bytes;
        live_.push_back({head_, serial});
        return begin % capacity_;
    }

private:
    struct Allocation { uint64_t end, serial; };
    uint64_t capacity_, head_ = 0, tail_ = 0;
    std::deque<Allocation> live_;
};

}
