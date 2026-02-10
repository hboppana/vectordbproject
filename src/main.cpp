#include "index/flat/flat_index.h"
#include <iostream>
#include <random>
#include <chrono>

Vector random_vector(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Vector v(dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

int main() {
    const size_t dim = 128;
    const size_t n = 10000; // 10,000 vectors
    const size_t k = 5; // return 5 closest vectors by euclidean distance

    FlatIndex index(dim);

    for (size_t i = 0; i < n; i++) {
        index.add(random_vector(dim));
    }

    Vector query = random_vector(dim);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = index.search(query, k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Query time: " << elapsed.count() << " ms\n"; // query time

    return 0;
}
