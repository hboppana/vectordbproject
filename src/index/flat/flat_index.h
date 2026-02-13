#pragma once
#include "index/index_base.h"

class FlatIndex : public IndexBase {
public:
    explicit FlatIndex(size_t dim); // fixed dim of vectors

    void add(const Vector& vec) override; // appends vector
    std::vector<size_t> search(const Vector& query, size_t k) const override; // returns indices of top-k nearest vectors
    size_t size() const override;

private:
    size_t dim_;
    std::vector<Vector> vectors_;
};