#include "src/core/distance.h"
#include "src/core/vector.h"
#include "src/utils/timer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

// Scalar version WITHOUT SIMD (4-way unroll only)
inline float l2_distance_scalar_only(const Vector& a, const Vector& b) {
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

Vector random_vector(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Vector v(dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

int main() {
    const size_t dim = 128;
    const size_t num_vectors = 10000;
    const size_t num_comparisons = 100000;

    std::cout << "\n=== SIMD vs Scalar Distance Comparison ===\n";
    std::cout << "Dimension: " << dim << "\n";
    std::cout << "Vectors: " << num_vectors << "\n";
    std::cout << "Comparisons: " << num_comparisons << "\n\n";

    // Generate test vectors
    std::cout << "Generating test vectors...\n";
    Vector query = random_vector(dim);
    std::vector<Vector> vectors;
    for (size_t i = 0; i < num_vectors; i++) {
        vectors.push_back(random_vector(dim));
    }

    // Warmup
    float dummy = 0.0f;
    for (size_t i = 0; i < 1000; i++) {
        dummy += l2_distance(query, vectors[i % num_vectors]);
    }

    // Test SIMD version (current l2_distance with AVX2)
    std::cout << "\n--- SIMD Version (with AVX2) ---\n";
    Timer simd_timer;
    simd_timer.reset();
    volatile float simd_result = 0.0f;
    for (size_t i = 0; i < num_comparisons; i++) {
        simd_result += l2_distance(query, vectors[i % num_vectors]);
    }
    double simd_time = simd_timer.elapsed_ms();
    std::cout << "Time: " << std::fixed << std::setprecision(3) << simd_time << " ms\n";
    std::cout << "Per-distance: " << (simd_time / num_comparisons * 1000.0) << " µs\n";

    // Test scalar version (4-way unroll only)
    std::cout << "\n--- Scalar Version (4-way unroll, no SIMD) ---\n";
    Timer scalar_timer;
    scalar_timer.reset();
    volatile float scalar_result = 0.0f;
    for (size_t i = 0; i < num_comparisons; i++) {
        scalar_result += l2_distance_scalar_only(query, vectors[i % num_vectors]);
    }
    double scalar_time = scalar_timer.elapsed_ms();
    std::cout << "Time: " << std::fixed << std::setprecision(3) << scalar_time << " ms\n";
    std::cout << "Per-distance: " << (scalar_time / num_comparisons * 1000.0) << " µs\n";

    // Calculate speedup
    double speedup = scalar_time / simd_time;
    double improvement = (scalar_time - simd_time) / scalar_time * 100.0;
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "Improvement: " << improvement << "%\n";

    return 0;
}
