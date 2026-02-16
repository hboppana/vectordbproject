#include "hnsw_index.h"
#include "core/distance.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

HNSWIndex::HNSWIndex(size_t dim, size_t M)
    : dim_(dim),
      M_(M),
      max_level_(0),
      entry_point_(0) {}

void HNSWIndex::set_ef_search(size_t ef_search) {
    ef_search_ = std::max<size_t>(1, ef_search);
}

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

std::vector<size_t> HNSWIndex::find_nearest_at_level(
    const Vector& query,
    size_t entry,
    int level,
    size_t M
) const {
    if (nodes_.empty()) {
        return {};
    }

    const size_t start = greedy_search(query, entry, level);
    std::vector<size_t> candidates;
    candidates.reserve(1 + (level < static_cast<int>(nodes_[start].neighbors.size())
        ? nodes_[start].neighbors[level].size() : 0));
    candidates.push_back(start);

    if (level < static_cast<int>(nodes_[start].neighbors.size())) {
        const auto& start_neighbors = nodes_[start].neighbors[level];
        candidates.insert(candidates.end(), start_neighbors.begin(), start_neighbors.end());
    }

    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(candidates.size());
    for (size_t idx : candidates) {
        float dist = l2_distance(query, nodes_[idx].vector);
        scored.emplace_back(dist, idx);
    }

    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    const size_t count = std::min(M, scored.size());
    std::vector<size_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count; i++) {
        result.push_back(scored[i].second);
    }
    return result;
}

// add node function
void HNSWIndex::add(const Vector& vec) {
    if (nodes_.empty()) {
        Node node;
        node.vector = vec;
        node.level = generate_level();
        node.neighbors.resize(node.level + 1);
        nodes_.push_back(node);
        entry_point_ = 0;
        max_level_ = node.level;
        return;
    }

    Node node;
    node.vector = vec;
    node.level = generate_level();
    node.neighbors.resize(node.level + 1);

    const size_t new_index = nodes_.size();
    nodes_.push_back(node);

    size_t current = entry_point_;
    for (int level = max_level_; level > node.level; level--) {
        current = greedy_search(vec, current, level);
    }

    for (int level = node.level; level >= 0; level--) {
        auto selected = find_nearest_at_level(vec, current, level, M_);
        nodes_[new_index].neighbors[level] = selected;

        for (size_t neighbor : selected) {
            if (level >= static_cast<int>(nodes_[neighbor].neighbors.size())) {
                nodes_[neighbor].neighbors.resize(level + 1);
            }
            nodes_[neighbor].neighbors[level].push_back(new_index);
        }
    }

    if (node.level > max_level_) {
        max_level_ = node.level;
        entry_point_ = new_index;
    }
}

// search function
std::vector<size_t> HNSWIndex::search(
    const Vector& query,
    size_t k
) const {
    if (nodes_.empty() || k == 0) {
        return {};
    }

    size_t current = entry_point_;

    for (int level = max_level_; level > 0; level--) {
        current = greedy_search(query, current, level);
    }

    current = greedy_search(query, current, 0);

    using DistId = std::pair<float, size_t>;

    auto min_heap_cmp = [](const DistId& a, const DistId& b) {
        return a.first > b.first;
    };

    std::priority_queue<DistId, std::vector<DistId>, decltype(min_heap_cmp)> candidates(min_heap_cmp);
    std::priority_queue<DistId> best_results;
    std::unordered_set<size_t> visited;
    visited.reserve(nodes_.size());

    const float entry_dist = l2_distance(query, nodes_[current].vector);
    candidates.emplace(entry_dist, current);
    best_results.emplace(entry_dist, current);
    visited.insert(current);

    size_t explored = 0;
    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_search_ && candidate_dist > best_results.top().first) {
            break;
        }

        if (explored++ >= ef_search_) {
            break;
        }

        if (0 < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[0]) {
                if (visited.find(neighbor) != visited.end()) {
                    continue;
                }

                visited.insert(neighbor);
                float dist = l2_distance(query, nodes_[neighbor].vector);
                candidates.emplace(dist, neighbor);
                best_results.emplace(dist, neighbor);

                if (best_results.size() > ef_search_) {
                    best_results.pop();
                }
            }
        }
    }

    std::vector<DistId> scored;
    scored.reserve(best_results.size());
    while (!best_results.empty()) {
        scored.push_back(best_results.top());
        best_results.pop();
    }

    std::sort(scored.begin(), scored.end(),
        [](const DistId& a, const DistId& b) {
            return a.first < b.first;
        });

    const size_t count = std::min(k, scored.size());
    std::vector<size_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count; i++) {
        result.push_back(scored[i].second);
    }

    return result;
}
