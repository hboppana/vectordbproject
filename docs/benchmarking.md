# Benchmarking Notes

## Hardware
- CPU: TBD
- RAM: TBD
- OS: Windows

## Dataset and Query Setup
- Dimensionality: 128
- Dataset type: random uniform floats in [0.0, 1.0]
- Index: Flat (exact L2)
- Queries per size: 100
- Reported metrics: average latency and p95 latency

## Results

| N (vectors) | Avg Latency (ms) | P95 Latency (ms) |
|-------------|------------------|-----------------|
| 10,000      | 10.2553 ms       | 13.8349 ms      |
| 50,000      | 54.2762 ms       | 66.1041 ms      |
| 100,000     | 113.8 ms         | 125.266 ms      |
| 250,000     | 260.814 ms       | 332.503 ms      |

## Observations
- Linear scaling: Latency grows roughly linearly with N, as expected for a flat index.
- Distance computation dominates: Most time is spent in L2 distance evaluation across all vectors.
- Exact search infeasible beyond X: For larger N, exact search becomes too slow for interactive latency targets O(N*d); use approximate indexing beyond TBD.

## Notes
- These numbers are single-run results; repeat runs can reduce noise.
- If you change the random seed or compiler flags, results will shift slightly.
