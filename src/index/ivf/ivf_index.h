#pragma once

#include "index/index_base.h"
#include <cstddef>
#include <vector>

class IVFIndex : public IndexBase {
public:
    IVFIndex(size_t dim, size_t nlist = 256, size_t nprobe = 8);

    // Train coarse centroids with k-means before adding vectors.
    void train(const std::vector<Vector>& training_data, size_t iterations = 12);

    void set_nprobe(size_t nprobe);
    bool is_trained() const;

    void add(const Vector& vec) override;
    std::vector<size_t> search(const Vector& query, size_t k) const override;
    size_t size() const override;

private:
    size_t nearest_centroid(const Vector& vec) const;

    size_t dim_;
    size_t nlist_;
    size_t nprobe_;
    bool trained_;

    std::vector<Vector> centroids_;
    std::vector<std::vector<size_t>> lists_;
    std::vector<Vector> vectors_;
};
