#include "index/flat/flat_index.h"
#include "utils/timer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

Vector random_vector(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Vector v(dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

int main() {
    const size_t dim = 128;
    const size_t k = 5; // return 5 closest vectors by euclidean distance
    const size_t queries = 100;
    const std::vector<size_t> sizes = {10000, 50000, 100000, 250000};

    for (size_t n : sizes) {
        FlatIndex index(dim);
        for (size_t i = 0; i < n; i++) {
            index.add(random_vector(dim));
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(queries);

        for (size_t i = 0; i < queries; i++) {
            Vector query = random_vector(dim);
            Timer timer;
            auto results = index.search(query, k);
            (void)results;
            latencies_ms.push_back(timer.elapsed_ms());
        }

        const double total = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
        const double avg = total / static_cast<double>(latencies_ms.size());

        std::sort(latencies_ms.begin(), latencies_ms.end());
        const size_t p95_index = static_cast<size_t>(std::ceil(0.95 * latencies_ms.size())) - 1;
        const double p95 = latencies_ms[p95_index];

        std::cout << "N=" << n << " -> avg " << avg << " ms, p95 " << p95 << " ms\n";
    }

    return 0;
}
