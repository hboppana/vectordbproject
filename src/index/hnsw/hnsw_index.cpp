#include "hnsw_index.h"
#include "core/distance.h"
#include <cmath>

HNSWIndex::HNSWIndex(size_t dim, size_t M)
    : dim_(dim),
      M_(M),
      max_level_(0),
      entry_point_(0) {}

size_t HNSWIndex::size() const {
    return nodes_.size();
}


int HNSWIndex::generate_level() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    double r = -std::log(dist(gen)) * 1.0;
    return static_cast<int>(r);
}


// basic greedy search
size_t HNSWIndex::greedy_search(
    const Vector& query,
    size_t entry,
    int level
) const {
    if (level >= static_cast<int>(nodes_[entry].neighbors.size()) ||
        nodes_[entry].neighbors[level].empty()) {
        return entry;
    }

    size_t current = entry;
    float current_dist = l2_distance(query, nodes_[current].vector);

    while (true) {
        bool improved = false;
        if (level < static_cast<int>(nodes_[current].neighbors.size())) {
            for (size_t neighbor : nodes_[current].neighbors[level]) {
                float dist = l2_distance(query, nodes_[neighbor].vector);
                if (dist < current_dist) {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }
        }

        if (!improved) {
            break;
        }
    }

    return current;
}

// add node function
void HNSWIndex::add(const Vector& vec) {
    Node node;
    node.vector = vec;
    node.level = generate_level();
    node.neighbors.resize(node.level + 1);

    nodes_.push_back(node);

    if (nodes_.size() == 1) {
        entry_point_ = 0;
        max_level_ = node.level;
    } else {
        if (node.level > max_level_) {
            max_level_ = node.level;
            entry_point_ = nodes_.size() - 1;
        }
    }
}

// search function
std::vector<size_t> HNSWIndex::search(
    const Vector& query,
    size_t k
) const {
    (void)k;

    if (nodes_.empty()) {
        return {};
    }

    size_t current = entry_point_;

    for (int level = max_level_; level > 0; level--) {
        current = greedy_search(query, current, level);
    }

    current = greedy_search(query, current, 0);

    return {current};
}
