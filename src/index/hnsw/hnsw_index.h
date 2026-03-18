#pragma once
#include "index/index_base.h"
#include <vector>
#include <random>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <memory>
#include <string>

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

    // Persistence: binary save/load of complete graph state.
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    int max_level() const { return max_level_; }

    void print_degree_stats() const;
    void reserve(size_t n);

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
    std::vector<std::unique_ptr<std::mutex>> node_mutexes_;
    std::atomic<size_t> node_count_{0};
    std::mutex entry_mutex_;

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
