#include "index/ivf/ivf_index.h"
#include "core/distance.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

IVFIndex::IVFIndex(size_t dim, size_t nlist, size_t nprobe)
    : dim_(dim),
      nlist_(std::max<size_t>(1, nlist)),
      nprobe_(std::max<size_t>(1, nprobe)),
      trained_(false) {}

void IVFIndex::train(const std::vector<Vector>& training_data, size_t iterations) {
    if (training_data.empty()) {
        trained_ = false;
        centroids_.clear();
        lists_.clear();
        return;
    }

    const size_t actual_nlist = std::min(nlist_, training_data.size());
    nlist_ = std::max<size_t>(1, actual_nlist);
    nprobe_ = std::min(nprobe_, nlist_);

    centroids_.clear();
    centroids_.reserve(nlist_);

    std::vector<size_t> sample_idx(training_data.size());
    std::iota(sample_idx.begin(), sample_idx.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(sample_idx.begin(), sample_idx.end(), rng);

    for (size_t i = 0; i < nlist_; ++i) {
        centroids_.push_back(training_data[sample_idx[i]]);
    }

    for (size_t it = 0; it < std::max<size_t>(1, iterations); ++it) {
        std::vector<Vector> sums(nlist_, Vector(dim_, 0.0f));
        std::vector<size_t> counts(nlist_, 0);

        for (const auto& vec : training_data) {
            if (vec.size() != dim_) {
                continue;
            }
            size_t cid = nearest_centroid(vec);
            counts[cid]++;
            for (size_t d = 0; d < dim_; ++d) {
                sums[cid][d] += vec[d];
            }
        }

        for (size_t c = 0; c < nlist_; ++c) {
            if (counts[c] == 0) {
                centroids_[c] = training_data[sample_idx[c % training_data.size()]];
                continue;
            }
            const float inv = 1.0f / static_cast<float>(counts[c]);
            for (size_t d = 0; d < dim_; ++d) {
                centroids_[c][d] = sums[c][d] * inv;
            }
        }
    }

    lists_.assign(nlist_, {});
    vectors_.clear();
    trained_ = true;
}

void IVFIndex::set_nprobe(size_t nprobe) {
    nprobe_ = std::max<size_t>(1, nprobe);
}

bool IVFIndex::is_trained() const {
    return trained_;
}

size_t IVFIndex::nearest_centroid(const Vector& vec) const {
    size_t best = 0;
    float best_dist = std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < centroids_.size(); ++i) {
        float dist = l2_distance(vec, centroids_[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }

    return best;
}

void IVFIndex::add(const Vector& vec) {
    if (!trained_ || vec.size() != dim_) {
        return;
    }

    const size_t idx = vectors_.size();
    vectors_.push_back(vec);

    size_t cid = nearest_centroid(vec);
    lists_[cid].push_back(idx);
}

std::vector<size_t> IVFIndex::search(const Vector& query, size_t k) const {
    if (!trained_ || query.size() != dim_ || k == 0 || vectors_.empty()) {
        return {};
    }

    std::vector<std::pair<float, size_t>> centroid_scores;
    centroid_scores.reserve(centroids_.size());

    for (size_t cid = 0; cid < centroids_.size(); ++cid) {
        float d = l2_distance(query, centroids_[cid]);
        centroid_scores.emplace_back(d, cid);
    }

    const size_t probes = std::min(nprobe_, centroid_scores.size());
    std::partial_sort(
        centroid_scores.begin(),
        centroid_scores.begin() + probes,
        centroid_scores.end());

    std::vector<std::pair<float, size_t>> candidates;
    for (size_t i = 0; i < probes; ++i) {
        const size_t cid = centroid_scores[i].second;
        for (size_t vid : lists_[cid]) {
            float d = l2_distance(query, vectors_[vid]);
            candidates.emplace_back(d, vid);
        }
    }

    if (candidates.empty()) {
        return {};
    }

    const size_t out_k = std::min(k, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + out_k,
        candidates.end());

    std::vector<size_t> result;
    result.reserve(out_k);
    for (size_t i = 0; i < out_k; ++i) {
        result.push_back(candidates[i].second);
    }

    return result;
}

size_t IVFIndex::size() const {
    return vectors_.size();
}
