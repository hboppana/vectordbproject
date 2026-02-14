#pragma once
#include "index/index_base.h"
#include <vector>
#include <random>

class HNSWIndex : public IndexBase {
public:
    HNSWIndex(size_t dim, size_t M = 16);

    void add(const Vector& vec) override;
    std::vector<size_t> search(
        const Vector& query,
        size_t k
    ) const override;

    size_t size() const override;

private:
    struct Node {
        Vector vector;
        int level;
        std::vector<std::vector<size_t>> neighbors; 
        // neighbors[level] = list of neighbor node indices
    };

    size_t dim_;
    size_t M_;                // max neighbors per level
    int max_level_;
    size_t entry_point_;      // index of entry node

    std::vector<Node> nodes_;

    int generate_level();

    // greedy search
    size_t greedy_search(
        const Vector& query,
        size_t entry,
        int level
    ) const;
};
