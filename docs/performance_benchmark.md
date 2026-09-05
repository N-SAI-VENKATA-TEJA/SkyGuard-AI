# SkyGuard AI — Performance Benchmark Report

## Latency (per observation)
| Metric | Value |
|---|---|
| Mean | 148.28 ms |
| Median | 146.89 ms |
| P95 | 170.02 ms |
| P99 | 183.16 ms |
| Max | 195.30 ms |
| Min | 124.75 ms |

## Throughput
- **6.7 observations/second**

## Memory
| Metric | Value |
|---|---|
| Pipeline init peak | 4134.1 KB |
| Runtime peak | 5.09 MB |

## Energy Estimate (Heuristic)
| Metric | Value |
|---|---|
| Assumed CPU TDP | 15W |
| Energy per observation | 1112.09 mJ |
| Total benchmark energy | 111.21 J |

## Edge Deployment Notes
- Memory footprint: 5.1 MB (exceeds ESP32 520KB SRAM — requires model distillation for edge)
- Inference latency: 148.3 ms (suitable for 10-minute AWS observation intervals)
- Recommendation: Deploy full pipeline on gateway/edge server; use lightweight statistical subset on ESP32
