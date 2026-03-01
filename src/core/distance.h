#pragma once
#include "vector.h"
#include <cmath>

// P4: Optimized euclidean distance with 4-way loop unrolling
inline float l2_distance(const Vector& a, const Vector& b) {
    const size_t n = a.size();
    const size_t n4 = n - (n % 4);
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (size_t i = 0; i < n4; i += 4) {
        float d0 = a[i]     - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (size_t i = n4; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}