#pragma once
#include "vector.h"
#include <cmath>

// euclidean distance between two vectors
inline float l2_distance(const Vector& a, const Vector& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}