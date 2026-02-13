#pragma once
#include "core/vector.h"
#include <vector>
#include <cstddef>

class IndexBase {
public:
    virtual ~IndexBase() = default;

    virtual void add(const Vector& vec) = 0;

    virtual std::vector<size_t> search(
        const Vector& query,
        size_t k
    ) const = 0;

    virtual size_t size() const = 0;
};
