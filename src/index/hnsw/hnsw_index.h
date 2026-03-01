#pragma once
#include "index/index_base.h"
#include <vector>
#include <random>
#include <cstdint>

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

    int max_level() const { return max_level_; }

    void print_degree_stats() const;

    // P2: Neighbor selection with keepPruned fallback
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
    size_t M_;                // max neighbors per level (higher levels)
    size_t M_max0_;           // max neighbors at level 0 = 2*M
    int max_level_ = 0;
    size_t entry_point_ = 0;  // index of entry node
    size_t ef_search_ = 20;   // for ef_search
    size_t ef_construction_ = 100; // P3: reduced from 200 to 100

    std::vector<Node> nodes_;

    // P0: Flat visited array with generation counter (avoids unordered_set)
    mutable std::vector<uint32_t> visited_marker_;
    mutable uint32_t visited_gen_ = 0;

    void reset_visited() const {
        ++visited_gen_;
        if (visited_gen_ == 0) {
            // Overflow: clear the whole array
            std::fill(visited_marker_.begin(), visited_marker_.end(), 0);
            visited_gen_ = 1;
        }
    }

    bool is_visited(size_t id) const {
        return id < visited_marker_.size() && visited_marker_[id] == visited_gen_;
    }

    void mark_visited(size_t id) const {
        if (id >= visited_marker_.size()) {
            visited_marker_.resize(id + 1, 0);
        }
        visited_marker_[id] = visited_gen_;
    }

    int random_level();

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
