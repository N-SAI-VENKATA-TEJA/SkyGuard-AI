"""
SkyGuard AI — Performance & Energy Benchmark
Measures CPU time, memory usage, and throughput per observation.
"""
import time
import tracemalloc
import numpy as np
import pandas as pd

from ml.anomaly_engine.stream_pipeline import StreamPipeline


def run_benchmark(n_observations=100):
    print("==================================================")
    print("SkyGuard AI — Performance & Energy Benchmark")
    print("==================================================\n")

    # Load sample data
    df = pd.read_csv('data/processed/aws_clean.csv', nrows=n_observations)
    print(f"Benchmarking with {len(df)} observations...\n")

    # 1. Memory Measurement
    tracemalloc.start()
    pipeline = StreamPipeline()
    _, peak_init = tracemalloc.get_traced_memory()
    print(f"[Memory] Pipeline initialization peak: {peak_init / 1024:.1f} KB")

    # 2. Latency Measurement
    latencies = []
    tracemalloc.reset_peak()

    for i, row in df.iterrows():
        t0 = time.perf_counter()
        pipeline.process_observation(
            timestamp=pd.Timestamp(row['timestamp']),
            temperature=row['temperature'],
            pressure=row['pressure'],
            humidity=row['humidity']
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    _, peak_runtime = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = np.array(latencies)

    # 3. Compute Statistics
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    max_lat = np.max(latencies)
    min_lat = np.min(latencies)
    throughput = 1000.0 / mean_lat  # obs/sec

    # 4. Energy Estimate (heuristic)
    # Modern laptop CPU TDP ~15W, assume 50% utilization during inference
    estimated_cpu_power_w = 15.0 * 0.5
    total_time_s = sum(latencies) / 1000.0
    energy_per_obs_mj = (estimated_cpu_power_w * (mean_lat / 1000.0)) * 1000  # millijoules

    # 5. Print Results
    print("\n--- LATENCY (per observation) ---")
    print(f"  Mean:     {mean_lat:.2f} ms")
    print(f"  Median:   {median_lat:.2f} ms")
    print(f"  P95:      {p95_lat:.2f} ms")
    print(f"  P99:      {p99_lat:.2f} ms")
    print(f"  Max:      {max_lat:.2f} ms")
    print(f"  Min:      {min_lat:.2f} ms")

    print("\n--- THROUGHPUT ---")
    print(f"  {throughput:.1f} observations/second")

    print("\n--- MEMORY ---")
    print(f"  Pipeline init peak:    {peak_init / 1024:.1f} KB")
    print(f"  Runtime peak:          {peak_runtime / 1024:.1f} KB ({peak_runtime / (1024*1024):.2f} MB)")

    print("\n--- ENERGY ESTIMATE (heuristic) ---")
    print(f"  Assumed CPU TDP:       {estimated_cpu_power_w * 2:.0f}W (50% utilization assumed)")
    print(f"  Energy per obs:        {energy_per_obs_mj:.2f} mJ")
    print(f"  Total benchmark time:  {total_time_s:.2f}s")
    print(f"  Total energy:          {estimated_cpu_power_w * total_time_s:.2f} J")

    print("\n--- EDGE DEPLOYMENT FEASIBILITY ---")
    # ESP32 runs at ~240 MHz, 520KB SRAM. Pipeline needs ~MB of memory.
    if peak_runtime / 1024 < 512:
        print("  Memory footprint is compatible with constrained devices.")
    else:
        print(f"  Memory footprint ({peak_runtime / (1024*1024):.1f} MB) exceeds ESP32 SRAM (520 KB).")
        print("  Recommendation: Use model distillation or edge-optimized inference.")

    if mean_lat < 1000:
        print(f"  Inference latency ({mean_lat:.1f} ms) is suitable for real-time 10-min AWS intervals.")
    else:
        print(f"  Inference latency ({mean_lat:.1f} ms) may need optimization for edge deployment.")

    # 6. Save report
    report = f"""# SkyGuard AI — Performance Benchmark Report

## Latency (per observation)
| Metric | Value |
|---|---|
| Mean | {mean_lat:.2f} ms |
| Median | {median_lat:.2f} ms |
| P95 | {p95_lat:.2f} ms |
| P99 | {p99_lat:.2f} ms |
| Max | {max_lat:.2f} ms |
| Min | {min_lat:.2f} ms |

## Throughput
- **{throughput:.1f} observations/second**

## Memory
| Metric | Value |
|---|---|
| Pipeline init peak | {peak_init / 1024:.1f} KB |
| Runtime peak | {peak_runtime / (1024*1024):.2f} MB |

## Energy Estimate (Heuristic)
| Metric | Value |
|---|---|
| Assumed CPU TDP | {estimated_cpu_power_w * 2:.0f}W |
| Energy per observation | {energy_per_obs_mj:.2f} mJ |
| Total benchmark energy | {estimated_cpu_power_w * total_time_s:.2f} J |

## Edge Deployment Notes
- Memory footprint: {peak_runtime / (1024*1024):.1f} MB (exceeds ESP32 520KB SRAM — requires model distillation for edge)
- Inference latency: {mean_lat:.1f} ms (suitable for 10-minute AWS observation intervals)
- Recommendation: Deploy full pipeline on gateway/edge server; use lightweight statistical subset on ESP32
"""
    with open('docs/performance_benchmark.md', 'w') as f:
        f.write(report)
    print("\nReport saved to docs/performance_benchmark.md")

    print("\n==================================================")
    print("BENCHMARK COMPLETE")
    print("==================================================")


if __name__ == '__main__':
    run_benchmark()
