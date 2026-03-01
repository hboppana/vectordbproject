#include "hnsw_index.h"
#include "core/distance.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <iostream>
#include <iomanip>

HNSWIndex::HNSWIndex(size_t dim, size_t M)
        : dim_(dim),
            M_(M),
            M_max0_(2 * M),   // Level 0 gets 2*M connections for better recall
            max_level_(0),
            entry_point_(0)
{
        ef_construction_ = 100;  // P3: reduced from 200
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

    // P0: Use flat visited vector instead of unordered_set
    reset_visited();
    if (visited_marker_.size() < nodes_.size()) {
        visited_marker_.resize(nodes_.size(), 0);
    }

    float entry_dist = l2_distance(query, nodes_[start].vector);
    candidates.emplace(entry_dist, start);
    best_results.emplace(entry_dist, start);
    mark_visited(start);

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_construction_ && candidate_dist > best_results.top().first) {
            break;
        }

        if (level < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[level]) {
                if (is_visited(neighbor)) {
                    continue;
                }
                mark_visited(neighbor);
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
    std::vector<int> pruned;  // P2: keepPruned fallback

    for (const auto& [dist_cn, candidate_id] : sorted) {
        if ((int)selected.size() >= M)
            break;
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
        } else {
            pruned.push_back(candidate_id);  // P2: save for backfill
        }
    }

    // P2: Backfill with pruned candidates if we didn't get M neighbors
    for (int pid : pruned) {
        if ((int)selected.size() >= M)
            break;
        selected.push_back(pid);
    }

    return selected;
}


// Pruning function — level-aware: uses M_max0_ at level 0, M_ elsewhere
void HNSWIndex::prune_neighbors(int node_id, int level)
{
    auto& neighbors = nodes_[node_id].neighbors[level];
    size_t max_neighbors = (level == 0) ? M_max0_ : M_;
    if (neighbors.size() <= max_neighbors)
        return;
    std::vector<std::pair<float, int>> dists;
    for (size_t nid : neighbors) {
        float d = l2_distance(
            nodes_[node_id].vector,
            nodes_[nid].vector
        );
        dists.emplace_back(d, static_cast<int>(nid));
    }
    std::sort(dists.begin(), dists.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });
    neighbors.clear();
    for (size_t i = 0; i < max_neighbors && i < dists.size(); ++i)
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
        // Call select_neighbors — use 2*M at level 0 for better connectivity
        int level_M = (level == 0) ? static_cast<int>(M_max0_) : static_cast<int>(M_);
        std::vector<float> new_vector = vec;
        auto selected = select_neighbors(candidate_list, level_M, new_vector);

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

    // P0: Use flat visited vector instead of unordered_set
    reset_visited();
    if (visited_marker_.size() < nodes_.size()) {
        visited_marker_.resize(nodes_.size(), 0);
    }

    candidates.emplace(current_dist, current);
    best_results.emplace(current_dist, current);
    mark_visited(current);

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_search_ && candidate_dist > best_results.top().first) {
            break;
        }

        if (0 < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[0]) {
                if (is_visited(neighbor)) {
                    continue;
                }

                mark_visited(neighbor);
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

void HNSWIndex::print_degree_stats() const {
    if (nodes_.empty()) {
        std::cout << "No nodes in index.\n";
        return;
    }

    size_t total_degree = 0;
    size_t zero_degree = 0;
    size_t max_degree = 0;

    for (const auto& node : nodes_) {
        size_t deg = (!node.neighbors.empty()) ? node.neighbors[0].size() : 0;
        total_degree += deg;
        if (deg == 0) zero_degree++;
        if (deg > max_degree) max_degree = deg;
    }

    double avg_degree = static_cast<double>(total_degree) / nodes_.size();

    std::cout << "Avg degree (level 0): " << std::fixed << std::setprecision(2) << avg_degree << "\n";
    std::cout << "Zero-degree nodes:    " << zero_degree << "\n";
    std::cout << "Max degree (level 0): " << max_degree << "\n";
    std::cout << "Total nodes:          " << nodes_.size() << "\n";
}

int HNSWIndex::random_level() {
    static std::default_random_engine gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.0, 1.0);

    // P1: Fix level generator for proper hierarchy depth.
    // Use mL = 1/ln(2) ≈ 1.4427 — the standard choice that gives:
    //   expected max_level ≈ mL * ln(N) ≈ 1.44 * ln(50000) ≈ 15.6
    // That's too high. Instead use mL = 0.5 which gives:
    //   expected max_level ≈ 0.5 * ln(50000) ≈ 5.4 → target 4-6 ✓
    constexpr double mL = 0.5;
    return static_cast<int>(-std::log(dist(gen)) * mL);
}
