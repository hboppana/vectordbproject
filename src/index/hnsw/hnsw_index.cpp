#include "hnsw_index.h"
#include "core/distance.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

constexpr char kMagic[4] = {'V', 'D', 'B', 'I'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kTypeHnsw = 2;

template <typename T>
bool write_pod(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_pod(std::ifstream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace

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

void HNSWIndex::reserve(size_t n) {
    nodes_.resize(n);
    node_mutexes_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        node_mutexes_[i] = std::make_unique<std::mutex>();
    }
}

size_t HNSWIndex::size() const {
    return node_count_.load();
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
    if (node_count_.load() == 0) {
        return {};
    }

    using DistId = std::pair<float, size_t>;
    auto min_heap_cmp = [](const DistId& a, const DistId& b) {
        return a.first > b.first;
    };

    size_t start = greedy_search(query, entry, level);
    std::priority_queue<DistId, std::vector<DistId>, decltype(min_heap_cmp)> candidates(min_heap_cmp);
    std::priority_queue<DistId> best_results;

    // Thread-local visited array for concurrent safety
    thread_local std::vector<uint32_t> tl_visited;
    thread_local uint32_t tl_gen = 0;
    ++tl_gen;
    if (tl_gen == 0) {
        std::fill(tl_visited.begin(), tl_visited.end(), 0);
        tl_gen = 1;
    }
    size_t n = nodes_.size();
    if (tl_visited.size() < n) {
        tl_visited.resize(n, 0);
    }

    float entry_dist = l2_distance(query, nodes_[start].vector);
    candidates.emplace(entry_dist, start);
    best_results.emplace(entry_dist, start);
    tl_visited[start] = tl_gen;

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_construction_ && candidate_dist > best_results.top().first) {
            break;
        }

        if (level < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[level]) {
                if (neighbor < tl_visited.size() && tl_visited[neighbor] == tl_gen) {
                    continue;
                }
                if (neighbor >= tl_visited.size()) {
                    tl_visited.resize(neighbor + 1, 0);
                }
                tl_visited[neighbor] = tl_gen;
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

// add node function — thread-safe with fine-grained locking
void HNSWIndex::add(const Vector& vec) {
    int node_level = random_level();

    // Atomically claim a slot
    size_t new_index = node_count_.fetch_add(1);

    // Initialize node data (exclusive slot, no lock needed)
    nodes_[new_index].vector = vec;
    nodes_[new_index].level = node_level;
    nodes_[new_index].neighbors.resize(node_level + 1);
    // Reserve capacity to prevent reallocations during concurrent reads
    for (int l = 0; l <= node_level; ++l) {
        size_t cap = (l == 0) ? M_max0_ : M_;
        nodes_[new_index].neighbors[l].reserve(cap * 2);
    }

    // Handle first node
    if (new_index == 0) {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        entry_point_ = 0;
        max_level_ = node_level;
        return;
    }

    // Snapshot entry point and max level
    size_t current;
    int cur_max_level;
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        current = entry_point_;
        cur_max_level = max_level_;
    }

    // Top-Down Descent (read-only, no locks)
    if (cur_max_level > node_level) {
        for (int level = cur_max_level; level > node_level; --level) {
            current = greedy_search(vec, current, level);
        }
    }

    // For each level, find neighbors and update edges
    int level_bound = std::min(node_level, cur_max_level);
    for (int level = level_bound; level >= 0; --level) {
        auto candidate_ids = find_nearest_at_level(vec, current, level, ef_construction_);
        std::vector<std::pair<float, int>> candidate_list;
        candidate_list.reserve(candidate_ids.size());
        for (size_t id : candidate_ids) {
            float dist = l2_distance(vec, nodes_[id].vector);
            candidate_list.emplace_back(dist, static_cast<int>(id));
        }

        int level_M = (level == 0) ? static_cast<int>(M_max0_) : static_cast<int>(M_);
        std::vector<float> new_vector = vec;
        auto selected = select_neighbors(candidate_list, level_M, new_vector);

        // Insert bidirectional edges with per-node locking (lock smaller id first)
        for (int neighbor_id : selected) {
            size_t u = std::min(new_index, static_cast<size_t>(neighbor_id));
            size_t v = std::max(new_index, static_cast<size_t>(neighbor_id));
            std::scoped_lock lock(*node_mutexes_[u], *node_mutexes_[v]);

            if (level >= static_cast<int>(nodes_[neighbor_id].neighbors.size())) {
                nodes_[neighbor_id].neighbors.resize(level + 1);
            }
            nodes_[new_index].neighbors[level].push_back(neighbor_id);
            nodes_[neighbor_id].neighbors[level].push_back(static_cast<size_t>(new_index));
            prune_neighbors(neighbor_id, level);
        }
        // Prune new node's neighbors for this level
        {
            std::lock_guard<std::mutex> lock(*node_mutexes_[new_index]);
            prune_neighbors(static_cast<int>(new_index), level);
        }
    }

    // Update entry point and max level
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        if (node_level > max_level_) {
            entry_point_ = new_index;
            max_level_ = node_level;
        }
    }
}

// search function
std::vector<size_t> HNSWIndex::search(
    const Vector& query,
    size_t k
) const {
    if (node_count_.load() == 0 || k == 0) {
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

    // Thread-local visited array
    thread_local std::vector<uint32_t> tl_visited_s;
    thread_local uint32_t tl_gen_s = 0;
    ++tl_gen_s;
    if (tl_gen_s == 0) {
        std::fill(tl_visited_s.begin(), tl_visited_s.end(), 0);
        tl_gen_s = 1;
    }
    if (tl_visited_s.size() < nodes_.size()) {
        tl_visited_s.resize(nodes_.size(), 0);
    }

    candidates.emplace(current_dist, current);
    best_results.emplace(current_dist, current);
    tl_visited_s[current] = tl_gen_s;

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate] = candidates.top();
        candidates.pop();

        if (best_results.size() >= ef_search_ && candidate_dist > best_results.top().first) {
            break;
        }

        if (0 < static_cast<int>(nodes_[candidate].neighbors.size())) {
            for (size_t neighbor : nodes_[candidate].neighbors[0]) {
                if (neighbor < tl_visited_s.size() && tl_visited_s[neighbor] == tl_gen_s) {
                    continue;
                }

                if (neighbor >= tl_visited_s.size()) {
                    tl_visited_s.resize(neighbor + 1, 0);
                }
                tl_visited_s[neighbor] = tl_gen_s;
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
    size_t count = node_count_.load();
    if (count == 0) {
        std::cout << "No nodes in index.\n";
        return;
    }

    size_t total_degree = 0;
    size_t zero_degree = 0;
    size_t max_degree = 0;

    for (size_t i = 0; i < count; ++i) {
        const auto& node = nodes_[i];
        size_t deg = (!node.neighbors.empty()) ? node.neighbors[0].size() : 0;
        total_degree += deg;
        if (deg == 0) zero_degree++;
        if (deg > max_degree) max_degree = deg;
    }

    double avg_degree = static_cast<double>(total_degree) / count;

    std::cout << "Avg degree (level 0): " << std::fixed << std::setprecision(2) << avg_degree << "\n";
    std::cout << "Zero-degree nodes:    " << zero_degree << "\n";
    std::cout << "Max degree (level 0): " << max_degree << "\n";
    std::cout << "Total nodes:          " << count << "\n";
}

int HNSWIndex::random_level() {
    thread_local std::default_random_engine gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(0.0, 1.0);

    // P1: Fix level generator for proper hierarchy depth.
    // Use mL = 1/ln(2) ≈ 1.4427 — the standard choice that gives:
    //   expected max_level ≈ mL * ln(N) ≈ 1.44 * ln(50000) ≈ 15.6
    // That's too high. Instead use mL = 0.5 which gives:
    //   expected max_level ≈ 0.5 * ln(50000) ≈ 5.4 → target 4-6 ✓
    constexpr double mL = 0.5;
    return static_cast<int>(-std::log(dist(gen)) * mL);
}

bool HNSWIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    const uint64_t dim_u64 = static_cast<uint64_t>(dim_);
    const uint64_t m_u64 = static_cast<uint64_t>(M_);
    const uint64_t mmax0_u64 = static_cast<uint64_t>(M_max0_);
    const int32_t max_level_i32 = static_cast<int32_t>(max_level_);
    const uint64_t entry_u64 = static_cast<uint64_t>(entry_point_);
    const uint64_t ef_search_u64 = static_cast<uint64_t>(ef_search_);
    const uint64_t ef_construction_u64 = static_cast<uint64_t>(ef_construction_);
    const uint64_t node_count_u64 = static_cast<uint64_t>(node_count_.load());

    out.write(kMagic, sizeof(kMagic));
    if (!out ||
        !write_pod(out, kVersion) ||
        !write_pod(out, kTypeHnsw) ||
        !write_pod(out, dim_u64) ||
        !write_pod(out, m_u64) ||
        !write_pod(out, mmax0_u64) ||
        !write_pod(out, max_level_i32) ||
        !write_pod(out, entry_u64) ||
        !write_pod(out, ef_search_u64) ||
        !write_pod(out, ef_construction_u64) ||
        !write_pod(out, node_count_u64)) {
        return false;
    }

    const size_t count = static_cast<size_t>(node_count_u64);
    if (count > nodes_.size()) {
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        const auto& node = nodes_[i];
        if (node.vector.size() != dim_) {
            return false;
        }

        const int32_t level_i32 = static_cast<int32_t>(node.level);
        const uint64_t levels_u64 = static_cast<uint64_t>(node.neighbors.size());

        if (!write_pod(out, level_i32)) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(node.vector.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!out || !write_pod(out, levels_u64)) {
            return false;
        }

        for (const auto& lvl_neighbors : node.neighbors) {
            const uint64_t n_u64 = static_cast<uint64_t>(lvl_neighbors.size());
            if (!write_pod(out, n_u64)) {
                return false;
            }
            for (size_t nid : lvl_neighbors) {
                const uint64_t id_u64 = static_cast<uint64_t>(nid);
                if (!write_pod(out, id_u64)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool HNSWIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[4] = {};
    uint32_t version = 0;
    uint32_t type = 0;
    uint64_t dim_u64 = 0;
    uint64_t m_u64 = 0;
    uint64_t mmax0_u64 = 0;
    int32_t max_level_i32 = 0;
    uint64_t entry_u64 = 0;
    uint64_t ef_search_u64 = 0;
    uint64_t ef_construction_u64 = 0;
    uint64_t node_count_u64 = 0;

    in.read(magic, sizeof(magic));
    if (!in ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !read_pod(in, version) ||
        !read_pod(in, type) ||
        !read_pod(in, dim_u64) ||
        !read_pod(in, m_u64) ||
        !read_pod(in, mmax0_u64) ||
        !read_pod(in, max_level_i32) ||
        !read_pod(in, entry_u64) ||
        !read_pod(in, ef_search_u64) ||
        !read_pod(in, ef_construction_u64) ||
        !read_pod(in, node_count_u64)) {
        return false;
    }

    if (version != kVersion || type != kTypeHnsw) {
        return false;
    }
    if (dim_u64 != static_cast<uint64_t>(dim_)) {
        return false;
    }
    if (m_u64 != static_cast<uint64_t>(M_) || mmax0_u64 != static_cast<uint64_t>(M_max0_)) {
        return false;
    }

    const size_t count = static_cast<size_t>(node_count_u64);
    if (count == 0) {
        nodes_.clear();
        node_mutexes_.clear();
        node_count_.store(0);
        max_level_ = 0;
        entry_point_ = 0;
        ef_search_ = static_cast<size_t>(ef_search_u64);
        ef_construction_ = static_cast<size_t>(ef_construction_u64);
        return true;
    }
    if (entry_u64 >= node_count_u64 || max_level_i32 < 0) {
        return false;
    }

    std::vector<Node> loaded_nodes;
    loaded_nodes.resize(count);

    for (size_t i = 0; i < count; ++i) {
        int32_t level_i32 = 0;
        uint64_t levels_u64 = 0;
        if (!read_pod(in, level_i32)) {
            return false;
        }
        if (level_i32 < 0) {
            return false;
        }

        loaded_nodes[i].level = static_cast<int>(level_i32);
        loaded_nodes[i].vector.resize(dim_);
        in.read(reinterpret_cast<char*>(loaded_nodes[i].vector.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!in || !read_pod(in, levels_u64)) {
            return false;
        }

        const size_t levels = static_cast<size_t>(levels_u64);
        if (levels != static_cast<size_t>(loaded_nodes[i].level + 1)) {
            return false;
        }

        loaded_nodes[i].neighbors.resize(levels);
        for (size_t l = 0; l < levels; ++l) {
            uint64_t n_u64 = 0;
            if (!read_pod(in, n_u64)) {
                return false;
            }
            const size_t n = static_cast<size_t>(n_u64);
            loaded_nodes[i].neighbors[l].resize(n);
            for (size_t j = 0; j < n; ++j) {
                uint64_t id_u64 = 0;
                if (!read_pod(in, id_u64) || id_u64 >= node_count_u64) {
                    return false;
                }
                loaded_nodes[i].neighbors[l][j] = static_cast<size_t>(id_u64);
            }
        }
    }

    std::vector<std::unique_ptr<std::mutex>> loaded_mutexes;
    loaded_mutexes.resize(count);
    for (size_t i = 0; i < count; ++i) {
        loaded_mutexes[i] = std::make_unique<std::mutex>();
    }

    nodes_.swap(loaded_nodes);
    node_mutexes_.swap(loaded_mutexes);
    node_count_.store(count);
    max_level_ = static_cast<int>(max_level_i32);
    entry_point_ = static_cast<size_t>(entry_u64);
    ef_search_ = static_cast<size_t>(ef_search_u64);
    ef_construction_ = static_cast<size_t>(ef_construction_u64);
    return true;
}
