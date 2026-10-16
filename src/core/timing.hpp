#pragma once

#include <chrono>

namespace sf {

using SteadyClock = std::chrono::steady_clock;

struct FrameHostTimings {
    double fence_wait_ms = 0.0;
    double acquire_wait_ms = 0.0;
    double submit_ms = 0.0;
    double present_ms = 0.0;
};

inline double elapsed_ms(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

class ScopedTimer {
public:
    explicit ScopedTimer(double* output) : output_(output), start_(output ? SteadyClock::now() : SteadyClock::time_point{}) {}
    ~ScopedTimer() { if (output_) *output_ = elapsed_ms(start_); }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double* output_;
    SteadyClock::time_point start_;
};

}
