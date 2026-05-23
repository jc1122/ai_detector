# Daemon Performance Report

## Command Used

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scripts/benchmark_daemon.py \
  --daemon-cmd ".venv/bin/python detector_daemon.py" \
  --workdir . \
  --text-size 3600 \
  --tmr-iterations 1 \
  --tmr-warmup 1 \
  --all-experts-iterations 1 \
  --all-experts-warmup 1 \
  --run-baseline true \
  --baseline-cold-runs 1 \
  --results-json reports/daemon_performance_results.json
```

## Run Summary

- TMR matrix: `3 (threads) × 3 (batch_size) × 2 (max_chunks) = 18` scenarios
- All-experts matrix: `1 × 1 × 1 = 1` scenario
- Measured iterations: `19` (18 TMR + 1 all-experts)
- Warmup iterations: `19` (18 TMR + 1 all-experts)
- Iteration total: `38` (`measured_total + warmup_total`)
- Baseline runs: `1` cold CLI run (`run_ensemble.py`)

## JSON Artifact Summary

- File: `reports/daemon_performance_results.json`
- Top-level fields now include:
  - `results`
  - `measured_total: 19`
  - `warmup_total: 19`
  - `iteration_total: 38`
  - `baseline` with `runs`, `latencies_ms`, and timing statistics

## Best Config / Timing

- Best TMR config by median latency: `tmr threads=4 batch=1 max_chunks=1`
- Median: **363.46 ms**
- p95: **363.46 ms**
- Startup: **2264.47 ms**
- Cold baseline (`run_ensemble.py`, 1 run): **7686.83 ms**
- All-experts config: `all experts threads=2 batch=4 max_chunks=1` median **4723.27 ms**

## Results (3600-char input, warmup excluded from latencies)

### TMR scenarios

| Scenario | Threads | Batch | MaxChunks | Startup ms | Median ms | p95 ms | Mean ms | Min ms | Max ms | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tmr threads=2 batch=1 max_chunks=1 | 2 | 1 | 1 | 2830.76 | 945.59 | 945.59 | 945.59 | 945.59 | 945.59 | n/a |
| tmr threads=2 batch=1 max_chunks=4 | 2 | 1 | 4 | 2761.39 | 999.66 | 999.66 | 999.66 | 999.66 | 999.66 | n/a |
| tmr threads=2 batch=4 max_chunks=1 | 2 | 4 | 1 | 2746.35 | 687.21 | 687.21 | 687.21 | 687.21 | 687.21 | n/a |
| tmr threads=2 batch=4 max_chunks=4 | 2 | 4 | 4 | 2702.96 | 1335.82 | 1335.82 | 1335.82 | 1335.82 | 1335.82 | n/a |
| tmr threads=2 batch=8 max_chunks=1 | 2 | 8 | 1 | 2304.56 | 601.29 | 601.29 | 601.29 | 601.29 | 601.29 | n/a |
| tmr threads=2 batch=8 max_chunks=4 | 2 | 8 | 4 | 2492.31 | 1388.57 | 1388.57 | 1388.57 | 1388.57 | 1388.57 | n/a |
| tmr threads=4 batch=1 max_chunks=1 | 4 | 1 | 1 | 2264.47 | 363.46 | 363.46 | 363.46 | 363.46 | 363.46 | n/a |
| tmr threads=4 batch=1 max_chunks=4 | 4 | 1 | 4 | 2168.97 | 538.15 | 538.15 | 538.15 | 538.15 | 538.15 | n/a |
| tmr threads=4 batch=4 max_chunks=1 | 4 | 4 | 1 | 2250.72 | 527.41 | 527.41 | 527.41 | 527.41 | 527.41 | n/a |
| tmr threads=4 batch=4 max_chunks=4 | 4 | 4 | 4 | 2406.48 | 740.19 | 740.19 | 740.19 | 740.19 | 740.19 | n/a |
| tmr threads=4 batch=8 max_chunks=1 | 4 | 8 | 1 | 2484.53 | 492.23 | 492.23 | 492.23 | 492.23 | 492.23 | n/a |
| tmr threads=4 batch=8 max_chunks=4 | 4 | 8 | 4 | 2525.64 | 756.93 | 756.93 | 756.93 | 756.93 | 756.93 | n/a |
| tmr threads=8 batch=1 max_chunks=1 | 8 | 1 | 1 | 2197.44 | 596.81 | 596.81 | 596.81 | 596.81 | 596.81 | n/a |
| tmr threads=8 batch=1 max_chunks=4 | 8 | 1 | 4 | 2384.01 | 864.06 | 864.06 | 864.06 | 864.06 | 864.06 | n/a |
| tmr threads=8 batch=4 max_chunks=1 | 8 | 4 | 1 | 2412.01 | 466.01 | 466.01 | 466.01 | 466.01 | 466.01 | n/a |
| tmr threads=8 batch=4 max_chunks=4 | 8 | 4 | 4 | 2586.70 | 1187.60 | 1187.60 | 1187.60 | 1187.60 | 1187.60 | n/a |
| tmr threads=8 batch=8 max_chunks=1 | 8 | 8 | 1 | 2384.69 | 474.71 | 474.71 | 474.71 | 474.71 | 474.71 | n/a |
| tmr threads=8 batch=8 max_chunks=4 | 8 | 8 | 4 | 2368.60 | 955.65 | 955.65 | 955.65 | 955.65 | 955.65 | n/a |

### All-experts scenario

| Scenario | Threads | Batch | MaxChunks | Startup ms | Median ms | p95 ms | Mean ms | Min ms | Max ms | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all experts threads=2 batch=4 max_chunks=1 | 2 | 4 | 1 | 3358.72 | 4723.27 | 4723.27 | 4723.27 | 4723.27 | 4723.27 | n/a |

## Residual Risks

- Peak RSS is unavailable in this run because resident-memory sampling fallback is not active in this environment (`peak_rss_mb` is `null` in results).
- The all-experts matrix is represented by a single measured point; any broad comparison should treat that as low-confidence for variance.
- Baseline and daemon measurements use single runs (`baseline_runs=1`, measured runs per scenario mostly one); this is enough for schema consistency but weak for statistical confidence.
