#include "index/flat/flat_index.h"
#include "core/distance.h"
#include <algorithm>

FlatIndex::FlatIndex(size_t dim) : dim_(dim) {}

void FlatIndex::add(const Vector& vec) {
    vectors_.push_back(vec);
}

std::vector<size_t> FlatIndex::search(const Vector& query, size_t k) const {
    std::vector<std::pair<float, size_t>> scores;
    k = std::min(k, scores.size());

    for (size_t i = 0; i < vectors_.size(); i++) {
        float dist = l2_distance(query, vectors_[i]);
        scores.emplace_back(dist, i);
    }

    std::partial_sort(
        scores.begin(),
        scores.begin() + k,
        scores.end()
    );

    std::vector<size_t> result;
    for (size_t i = 0; i < k; i++) {
        result.push_back(scores[i].second);
    }
    return result;
}
