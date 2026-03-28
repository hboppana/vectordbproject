# IVF Benchmark Report

## Purpose
This file records measured IVF MVP (IVF-Flat) performance and compares it with HNSW from the same benchmark run.

## Build And Run Command
```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first
.\build\vector_db.exe
```

## Run Metadata
- Date: 2026-03-28
- Project: vectordbproject
- Build system: CMake + MinGW Makefiles
- Dataset size: 50,000 vectors
- Dimension: 128
- Query count: 100

## Shared Benchmark Context
- Ground truth for recall is brute-force nearest neighbor.
- HNSW parameters: M=32, efConstruction=100, M_max0=64.
- IVF parameters: nlist=256, k-means iterations=12, nprobe sweep=4/8/16/32.

## HNSW Results (Same Run)

### Index Build
- Build time: 4.14s (PASS < 20s)
- Max level: 5 (within expected range 4-6)
- Avg degree (level 0): 63.94
- Zero-degree nodes: 0
- Max degree (level 0): 64
- Persistence round trip: save/load OK

### Search Sweep
| efSearch | Recall@1 | AvgMs | Status |
|---------:|---------:|------:|--------|
| 50       | 0.7200   | 0.7375 | Recall<0.85, Latency OK |
| 75       | 0.7800   | 0.9490 | Recall<0.85, Latency OK |
| 100      | 0.8400   | 1.5351 | Recall<0.85, Latency OK |
| 150      | 0.9000   | 1.7394 | Recall>=0.85, Latency OK |

## IVF MVP Results (Part 7)

### IVF Build
- Train time: 1.15s
- Add time: 0.11s
- Indexed vectors: 50,000

### nprobe Sweep
| nlist | nprobe | Recall@1 | AvgMs | Status |
|------:|-------:|---------:|------:|--------|
| 256   | 4      | 0.0900   | 0.0660 | Recall<0.85, Latency OK |
| 256   | 8      | 0.1700   | 0.1376 | Recall<0.85, Latency OK |
| 256   | 16     | 0.2800   | 0.2879 | Recall<0.85, Latency OK |
| 256   | 32     | 0.4300   | 0.5337 | Recall<0.85, Latency OK |

## Takeaways
- IVF MVP is very fast in this configuration (all tested nprobe values are sub-1ms avg query latency).
- Recall is currently much lower than HNSW for this dataset and setup.
- Recall improves as nprobe increases, which matches expected IVF behavior.

## Next Tuning Pass
- Test larger nprobe values (64, 96, 128) to map recall-latency curve further.
- Test larger nlist values (512 and 1024) to improve coarse partition quality.
- Increase k-means training quality (more iterations and/or larger training sample).
- Add centroid initialization improvements (for example, k-means++) and compare impact.
