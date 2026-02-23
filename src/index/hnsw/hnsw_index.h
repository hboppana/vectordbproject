#pragma once
#include "index/index_base.h"
#include <vector>
#include <random>

class HNSWIndex : public IndexBase {
public:
    HNSWIndex(size_t dim, size_t M = 16);

    void set_ef_search(size_t ef_search);

    void add(const Vector& vec) override;
    std::vector<size_t> search(
        const Vector& query,
        size_t k
    ) const override;

    size_t size() const override;

    // Clean neighbor selection function
    std::vector<int> select_neighbors(
        const std::vector<std::pair<float, int>>& candidates,
        int M,
        const std::vector<float>& new_vector);

    void prune_neighbors(int node_id, int level);

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
    size_t ef_search_ = 20; // for ef_search
    size_t ef_construction_ = 100; // start bigger than ef_search

    std::vector<Node> nodes_;

    int generate_level();

    // greedy search
    size_t greedy_search(
        const Vector& query,
        size_t entry,
        int level
    ) const;

    std::vector<size_t> find_nearest_at_level(
        const Vector& query,
        size_t entry,
        int level,
        size_t M
    ) const;
};
