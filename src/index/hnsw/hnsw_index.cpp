#include "hnsw_index.h"
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
    return {};
}
