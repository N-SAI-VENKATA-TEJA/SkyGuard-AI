# SIH26073 Requirement Matrix

| SIH Requirement | Implementation | Evidence | Status | Limitation |
|---|---|---|---|---|
| Real-time Anomaly Detection | FastAPI + StreamPipeline + Hybrid Engine | Measured latency ~83.36ms (API) | SUPPORTED | In-memory state limits horizontal scaling. |
| Anomalies in T, P, RH | Multivariate feature processing | Synthetic detection validation | SUPPORTED | T/P/RH explicitly cross-validated. |
| Spike/Transient Detection | Isolation Forest Density | Validated in E2E Tests | SUPPORTED | N/A |
| Persistent/Drift Detection | Rolling Statistical Derivations | Validated via `docs/step11_demo_evidence.md` | SUPPORTED | N/A |
| Frozen/Stuck Sensor | Strict Variance Monitoring | Validated via `test_step9_end_to_end.py` | SUPPORTED | N/A |
| Sensor Noise | N/A | Mild noise natively evades boundaries | LIMITATION | Smooths into natural envelope distributions. |
| Offset | Dynamic Context tracking | Validated in Step 9.2 experiments | SUPPORTED | N/A |
| Multivariate Inconsistency | Z-score divergence computations | Validated in Step 9.2 experiments | SUPPORTED | N/A |
| Severity Grading | Binned threshold outputs | Visualized on Dashboard | SUPPORTED | N/A |
| Confidence Approximation | Evidence Domain Agreement | Computed deterministically | SUPPORTED | Heuristic score, not statistically calibrated probability. |
| Deterministic Explanation | Rule-based mapping of ML outputs | Available on WebSocket stream | SUPPORTED | N/A |
| Sensor Health Degradation | Exponential streak penalty tracker | Validated via `test_step7_sensor_health.py` | SUPPORTED | N/A |
| Maintenance Recommendations | Bound to health states | Integrated into JSON response | SUPPORTED | N/A |
| Data Loss / Comm. Errors | Validated null interception routing | Verified in API contract testing | SUPPORTED | N/A |
| Imputed Corrected Values | Internal pipeline pre-processing | No explicit stream produced | OPTIONAL / NOT IMPLEMENTED | Pipeline survives missing data, but user UI feed remains missing. |
| Neighboring-Station Spatial | Station telemetry processed alone | Single-station isolated dictionaries | OPTIONAL / NOT IMPLEMENTED | System only tests intra-station multivariate consistency. |
| Dashboard Visualization | React + WebSockets | Validated manually | SUPPORTED | Browser E2E automated test latency unmeasured. |
| Multi-Station Scale | In-memory dynamic dicts | Validated via interleaving requests | PARTIAL | Stateful memory resets on server reboot. |
| Energy Efficiency | Software deployment | No hardware benchmarks conducted | NOT MEASURED | N/A |
