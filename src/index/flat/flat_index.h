#pragma once
#include "core/vector.h"
#include <vector>

class FlatIndex {
public:
    explicit FlatIndex(size_t dim); // fixed dim of vectors

    void add(const Vector& vec); // appends vector
    std::vector<size_t> search(const Vector& query, size_t k) const; // returns indices of top-k nearest vectors

private:
    size_t dim_;
    std::vector<Vector> vectors_;
};