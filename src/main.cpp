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
    const std::vector<size_t> ef_values = {10, 20, 50};

    for (size_t n : sizes) {
        HNSWIndex index(dim);
        std::vector<Vector> dataset;
        dataset.reserve(n);

        for (size_t i = 0; i < n; i++) {
            Vector v = random_vector(dim);
            dataset.push_back(v);
            index.add(v);
        }

        std::vector<Vector> query_set;
        query_set.reserve(queries);
        for (size_t q = 0; q < queries; q++) {
            query_set.push_back(random_vector(dim));
        }

        for (size_t ef : ef_values) {
            index.set_ef_search(ef);

            size_t matches = 0;
            double total_search_ms = 0.0;

            for (size_t q = 0; q < queries; q++) {
                const Vector& query = query_set[q];

                Timer search_timer;
                auto results = index.search(query, 1);
                total_search_ms += search_timer.elapsed_ms();

                if (results.empty()) {
                    continue;
                }

                const size_t returned = results[0];

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
            }

            const double recall = static_cast<double>(matches) / static_cast<double>(queries);
            const double avg_search_ms = total_search_ms / static_cast<double>(queries);

            std::cout << "N=" << n
                      << " ef=" << ef
                      << " -> Recall@1=" << recall
                      << ", AvgSearchMs=" << avg_search_ms
                      << "\n";
        }
    }

    // Previous benchmark logic can be re-enabled if needed.

    return 0;
}
