#include "hnsw_index.h"
#include "core/distance.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <iostream>

HNSWIndex::HNSWIndex(size_t dim, size_t M)
        : dim_(dim),
            M_(M),
            max_level_(0),
            entry_point_(0)
{
        ef_construction_ = 200;
        ef_search_ = 50;
}

void HNSWIndex::set_ef_search(size_t ef_search) {
    ef_search_ = std::max<size_t>(1, ef_search);
}

size_t HNSWIndex::size() const {
    return nodes_.size();
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

    using DistId = std::pair<float, size_t>;
    auto min_heap_cmp = [](const DistId& a, const DistId& b) {
        return a.first > b.first;
    };

    size_t start = greedy_search(query, entry, level);
    std::priority_queue<DistId, std::vector<DistId>, decltype(min_heap_cmp)> candidates(min_heap_cmp);
    std::priority_queue<DistId> best_results;
    std::unordered_set<size_t> visited;
    visited.reserve(nodes_.size());

    float entry_dist = l2_distance(query, nodes_[start].vector);
    candidates.emplace(entry_dist, start);
    best_results.emplace(entry_dist, start);
    visited.insert(start);

    size_t explored = 0;
    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_construction_ && candidate_dist > best_results.top().first) {
            break;
        }
        if (explored++ >= ef_construction_) {
            break;
        }

        if (level < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[level]) {
                if (visited.find(neighbor) != visited.end()) {
                    continue;
                }
                visited.insert(neighbor);
                float dist = l2_distance(query, nodes_[neighbor].vector);
                candidates.emplace(dist, neighbor);
                best_results.emplace(dist, neighbor);
                if (best_results.size() > ef_construction_) {
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

    const size_t count = std::min(M, scored.size());
    std::vector<size_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count; i++) {
        result.push_back(scored[i].second);
    }
    return result;
}

// Clean neighbor selection function
std::vector<int> HNSWIndex::select_neighbors(
    const std::vector<std::pair<float, int>>& candidates,
    int M,
    const std::vector<float>& new_vector)
{
    std::vector<std::pair<float, int>> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first; // sort by distance ascending
              });

    std::vector<int> selected;

    for (const auto& [dist_cn, candidate_id] : sorted) {
        bool good = true;
        for (int selected_id : selected) {
            float dist_cs = l2_distance(
                nodes_[candidate_id].vector,
                nodes_[selected_id].vector
            );
            if (dist_cs <= dist_cn) {
                good = false;
                break;
            }
        }
        if (good) {
            selected.push_back(candidate_id);
        }
        if ((int)selected.size() == M)
            break;
    }
    return selected;
}


// Simple pruning function
void HNSWIndex::prune_neighbors(int node_id, int level)
{
    auto& neighbors = nodes_[node_id].neighbors[level];
    if ((int)neighbors.size() <= M_)
        return;
    std::vector<std::pair<float, int>> dists;
    for (int nid : neighbors) {
        float d = l2_distance(
            nodes_[node_id].vector,
            nodes_[nid].vector
        );
        dists.emplace_back(d, nid);
    }
    std::sort(dists.begin(), dists.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });
    neighbors.clear();
    for (int i = 0; i < M_ && i < (int)dists.size(); ++i)
        neighbors.push_back(dists[i].second);
}

// add node function
void HNSWIndex::add(const Vector& vec) {
    int node_level = random_level();
    Node node;
    node.vector = vec;
    node.level = node_level;
    node.neighbors.resize(node_level + 1);

    const size_t new_index = nodes_.size();

    if (nodes_.empty()) {
        nodes_.push_back(node);
        entry_point_ = 0;
        max_level_ = node_level;
        return;
    }

    nodes_.push_back(node);

    size_t current = entry_point_;
    // Top-Down Descent: from max_level_ down to node_level+1 (exclusive)
    if (max_level_ > node_level) {
        for (int level = max_level_; level > node_level; --level) {
            current = greedy_search(vec, current, level);
        }
    }

    // For each level ≤ node_level (from min(node_level, max_level_) down to 0)
    int level_bound = std::min(node_level, max_level_);
    for (int level = level_bound; level >= 0; --level) {
        // Get candidate neighbors from efConstruction search
        auto candidate_ids = find_nearest_at_level(vec, current, level, ef_construction_);
        std::vector<std::pair<float, int>> candidate_list;
        candidate_list.reserve(candidate_ids.size());
        for (size_t id : candidate_ids) {
            float dist = l2_distance(vec, nodes_[id].vector);
            candidate_list.emplace_back(dist, static_cast<int>(id));
        }
        // Call select_neighbors
        std::vector<float> new_vector = vec; // Assuming Vector is std::vector<float>
        auto selected = select_neighbors(candidate_list, static_cast<int>(M_), new_vector);

        // Insert neighbors
        for (int neighbor_id : selected) {
            if (level >= static_cast<int>(nodes_[neighbor_id].neighbors.size())) {
                nodes_[neighbor_id].neighbors.resize(level + 1);
            }
            nodes_[new_index].neighbors[level].push_back(neighbor_id);
            nodes_[neighbor_id].neighbors[level].push_back(static_cast<int>(new_index));
            prune_neighbors(neighbor_id, level);
        }
        prune_neighbors(static_cast<int>(new_index), level);
    }

    // Update entry point and max_level_
    if (node_level > max_level_) {
        entry_point_ = static_cast<int>(new_index);
        max_level_ = node_level;
    }

    // Temporary debug prints for degree statistics
    int total = 0;
    int zero = 0;
    for (auto& node : nodes_) {
        int deg = node.neighbors.size() > 0 ? node.neighbors[0].size() : 0;
        total += deg;
        if (deg == 0) zero++;
    }
    std::cout << "AvgDeg=" << (float)total / nodes_.size()
              << " ZeroDeg=" << zero << std::endl;
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
    float current_dist = l2_distance(query, nodes_[current].vector);

    // Greedy top-down descent: settle at each level before moving down
    for (int level = max_level_; level > 0; level--) {
        bool improved = true;
        while (improved) {
            improved = false;
            if (level < static_cast<int>(nodes_[current].neighbors.size())) {
                for (size_t neighbor : nodes_[current].neighbors[level]) {
                    float neighbor_dist = l2_distance(query, nodes_[neighbor].vector);
                    if (neighbor_dist < current_dist) {
                        current = neighbor;
                        current_dist = neighbor_dist;
                        improved = true;
                    }
                }
            }
        }
    }

    // Level 0: full efSearch expansion
    using DistId = std::pair<float, size_t>;

    auto min_heap_cmp = [](const DistId& a, const DistId& b) {
        return a.first > b.first;
    };

    std::priority_queue<DistId, std::vector<DistId>, decltype(min_heap_cmp)> candidates(min_heap_cmp);
    std::priority_queue<DistId> best_results;
    std::unordered_set<size_t> visited;
    visited.reserve(nodes_.size());

    candidates.emplace(current_dist, current);
    best_results.emplace(current_dist, current);
    visited.insert(current);

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_search_ && candidate_dist > best_results.top().first) {
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

int HNSWIndex::random_level() {
    static std::default_random_engine gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Standard HNSW exponential decay: P(L >= l) = e^(-l * ln(M))
    // Equivalently: level = floor(-ln(rand) / ln(M))
    return static_cast<int>(-std::log(dist(gen)) / std::log(static_cast<float>(M_)));
}
