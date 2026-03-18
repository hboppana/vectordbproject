#pragma once
#include "vector.h"
#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics

// ============================================================
// SIMD L2 Distance Implementation with Fallback
// ============================================================

namespace simd {

// Horizontal sum of 8 floats in a __m256
inline float horizontal_sum_avx2(__m256 v) {
    __m128 upper = _mm256_extractf128_ps(v, 1);
    __m128 lower = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(upper, lower);
    
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sum64 = _mm_add_ps(sum128, shuf);
    shuf = _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 0, 3, 2));
    return _mm_cvtss_f32(_mm_add_ps(sum64, shuf));
}

// AVX2 optimized L2 distance (8 floats per iteration)
inline float l2_distance_avx2(const Vector& a, const Vector& b) {
    const size_t n = a.size();
    const size_t n8 = n - (n % 8);
    __m256 sum_vec = _mm256_setzero_ps();

    // Process 8 floats at a time with AVX2
    for (size_t i = 0; i < n8; i += 8) {
        __m256 a_vals = _mm256_loadu_ps(&a[i]);
        __m256 b_vals = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(a_vals, b_vals);
        __m256 squared = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, squared);
    }

    float sum = horizontal_sum_avx2(sum_vec);

    // Handle remainder (0-7 elements) with scalar code
    for (size_t i = n8; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }

    return sum;
}

// Scalar fallback (4-way unrolled)
inline float l2_distance_scalar(const Vector& a, const Vector& b) {
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

}  // namespace simd

// Select implementation: AVX2 or scalar fallback
inline float l2_distance(const Vector& a, const Vector& b) {
#ifdef __AVX2__
    return simd::l2_distance_avx2(a, b);
#else
    return simd::l2_distance_scalar(a, b);
#endif
}
