#pragma once

#include <chrono>

class Timer {
public:
    Timer() : start_(clock::now()) {}

    void reset() { start_ = clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - start_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};
