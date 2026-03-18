#include "core/distance.h"
#include "index/flat/flat_index.h"
#include "index/hnsw/hnsw_index.h"
#include "utils/timer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

Vector random_vector(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Vector v(dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

int main() {
    // ================================================================
    //  HNSW Optimization Benchmark
    //  N=50k, M=32, efConstruction=200
    // ================================================================
    const size_t dim     = 128;
    const int    N       = 50000;
    const int    queries = 100;
    const size_t M       = 32;
    const std::vector<size_t> ef_values = {50, 75, 100, 150};

    std::cout << "\n  DAY 13: HNSW Optimization Benchmark\n";
    std::cout << "N=" << N << "  dim=" << dim
              << "  M=" << M << "  efConstruction=100  M_max0=" << (2*M) << "\n\n";

    // ----------------------------------------------------------
    //  PART 1 + PART 4 — Build index, measure build time
    // ----------------------------------------------------------
    std::cout << "--- PART 1 & 4: Build Index (Multi-Threaded) ---\n";
    HNSWIndex index(dim, M);
    index.reserve(N);
    std::vector<Vector> dataset;
    dataset.reserve(N);

    // Pre-generate all vectors for parallel insertion
    std::cout << "  Generating " << N << " random vectors...\n";
    for (int i = 0; i < N; i++) {
        dataset.push_back(random_vector(dim));
    }

    // Parallel insertion using OpenMP
    const int num_threads = omp_get_num_procs();
    std::cout << "  Inserting with " << num_threads << " threads...\n";
    omp_set_num_threads(num_threads);
    
    Timer build_timer;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        index.add(dataset[i]);
    }
    
    double build_time_s = build_timer.elapsed_ms() / 1000.0;

    std::cout << "\nBuild time:  " << std::fixed << std::setprecision(2)
              << build_time_s << "s";
    if (build_time_s < 20.0)
        std::cout << "  [PASS < 20s]";
    else
        std::cout << "  [WARN >= 20s — investigate]";
    std::cout << "\n";

    // ----------------------------------------------------------
    //  PART 1 (cont) — Hierarchy depth
    // ----------------------------------------------------------
    std::cout << "\n--- Hierarchy Depth ---\n";
    std::cout << "Max level: " << index.max_level() << std::endl;
    if (index.max_level() >= 4 && index.max_level() <= 6)
        std::cout << "  [OK — expected 4-6]\n";
    else if (index.max_level() <= 3)
        std::cout << "  [LOW — level generator may be too steep]\n";
    else
        std::cout << "  [HIGH — possible bug in level generation]\n";

    // ----------------------------------------------------------
    //  PART 3 — Degree statistics
    // ----------------------------------------------------------
    std::cout << "\n--- PART 3: Degree Statistics (Level 0) ---\n";
    index.print_degree_stats();

    // ----------------------------------------------------------
    //  PART 6 — Persistence round-trip check
    // ----------------------------------------------------------
    std::cout << "\n--- PART 6: Persistence (Save/Load) ---\n";
    const std::string hnsw_path = "build/hnsw_index.bin";
    if (index.save(hnsw_path)) {
        HNSWIndex loaded_index(dim, M);
        if (loaded_index.load(hnsw_path)) {
            std::cout << "Saved + loaded index: OK\n";
            std::cout << "Loaded nodes: " << loaded_index.size()
                      << "  maxLevel: " << loaded_index.max_level() << "\n";
        } else {
            std::cout << "Load failed for " << hnsw_path << "\n";
        }
    } else {
        std::cout << "Save failed for " << hnsw_path << "\n";
    }

    // ----------------------------------------------------------
    //  Generate queries + brute-force ground truth
    // ----------------------------------------------------------
    std::cout << "\nGenerating " << queries << " queries + brute-force ground truth...\n";
    std::vector<Vector> query_set;
    query_set.reserve(queries);
    for (int q = 0; q < queries; q++) {
        query_set.push_back(random_vector(dim));
    }

    std::vector<size_t> ground_truth(queries);
    for (int q = 0; q < queries; q++) {
        size_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < dataset.size(); i++) {
            float d = l2_distance(query_set[q], dataset[i]);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        ground_truth[q] = best_idx;
    }
    std::cout << "Ground truth computed.\n";

    // ----------------------------------------------------------
    //  PART 2 + PART 5 — Recall sweep + search latency
    // ----------------------------------------------------------
    std::cout << "\n--- PART 2 & 5: Recall Sweep + Search Latency ---\n";
    std::cout << std::left
              << std::setw(12) << "efSearch"
              << std::setw(14) << "Recall@1"
              << std::setw(14) << "AvgMs"
              << "Status\n";
    std::cout << std::string(52, '-') << "\n";

    for (size_t ef : ef_values) {
        index.set_ef_search(ef);

        size_t matches = 0;
        double total_search_ms = 0.0;

        for (int q = 0; q < queries; q++) {
            Timer search_timer;
            auto results = index.search(query_set[q], 1);
            total_search_ms += search_timer.elapsed_ms();

            if (!results.empty() && results[0] == ground_truth[q]) {
                matches++;
            }
        }

        double recall = static_cast<double>(matches) / queries;
        double avg_ms = total_search_ms / queries;

        // Status string
        std::string status;
        if (recall >= 0.85) status += "Recall>=0.85 ";
        else                status += "Recall<0.85  ";
        if (ef == 50  && avg_ms < 2.0) status += "Latency OK";
        else if (ef == 75  && avg_ms < 3.0) status += "Latency OK";
        else if (ef == 100 && avg_ms < 4.0) status += "Latency OK";
        else if (ef == 150 && avg_ms < 6.0) status += "Latency OK";
        else status += "Latency HIGH";

        std::cout << std::left
                  << std::setw(12) << ef
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << recall
                  << std::setw(14) << avg_ms
                  << status << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "  BENCHMARK COMPLETE\n";
    std::cout << "========================================\n";

    return 0;
}
