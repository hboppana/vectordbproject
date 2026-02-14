#include "core/distance.h"
#include "index/flat/flat_index.h"
#include "index/hnsw/hnsw_index.h"
#include "utils/timer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
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
    const std::vector<size_t> sizes = {100, 1000, 10000};
    const size_t queries = 100;

    for (size_t n : sizes) {
        HNSWIndex index(dim);
        std::vector<Vector> dataset;
        dataset.reserve(n);

        for (size_t i = 0; i < n; i++) {
            Vector v = random_vector(dim);
            dataset.push_back(v);
            index.add(v);
        }

        size_t matches = 0;
        size_t last_returned = 0;
        float last_returned_dist = 0.0f;
        size_t last_brute_index = 0;
        float last_brute_dist = 0.0f;

        for (size_t q = 0; q < queries; q++) {
            Vector query = random_vector(dim);
            auto results = index.search(query, 1);

            if (results.empty()) {
                continue;
            }

            const size_t returned = results[0];
            const float returned_dist = l2_distance(query, dataset[returned]);

            size_t brute_index = 0;
            float brute_dist = std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < dataset.size(); i++) {
                float dist = l2_distance(query, dataset[i]);
                if (dist < brute_dist) {
                    brute_dist = dist;
                    brute_index = i;
                }
            }

            if (returned == brute_index) {
                matches++;
            }

            last_returned = returned;
            last_returned_dist = returned_dist;
            last_brute_index = brute_index;
            last_brute_dist = brute_dist;
        }

        const double recall = static_cast<double>(matches) / static_cast<double>(queries);

        std::cout << "N=" << n
                  << " -> returned " << last_returned
                  << " (dist " << last_returned_dist << ")"
                  << ", brute " << last_brute_index
                  << " (dist " << last_brute_dist << ")"
                  << ", Recall@1 = " << recall << "\n";
    }

    // Previous benchmark logic can be re-enabled if needed.

    return 0;
}
