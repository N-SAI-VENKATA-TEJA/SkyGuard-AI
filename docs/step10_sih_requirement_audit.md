# STEP 10: SIH26073 REQUIREMENT AUDIT

## 1. REAL-TIME ANOMALY DETECTION
- **Official Requirement**: Real-time anomaly detection for AWS observations.
- **Audit**: An observation successfully enters via a FastAPI POST endpoint, processes chronologically through an in-memory `StreamPipeline` feature engineer, evaluates via a frozen Hybrid V2 anomaly model, and broadcasts via a WebSocket connection to the frontend dashboard.
- **Latency Evidence**: 
  - `StreamPipeline` Benchmark: mean ≈47.11 ms
  - API Benchmark: mean ≈83.36 ms
  - Browser/UI E2E Latency: NOT MEASURED.
- **Status**: SUPPORTED. (Note: "Instantaneous" claims must be avoided; actual pipeline latency is ~80ms).

## 2. ANOMALY TYPES / SENSOR FAULTS
| Fault Type | Status | Evidence Source |
|---|---|---|
| Spike | SUPPORTED | Step 9 E2E (Synthetic Spike correctly triggered anomaly) |
| Drift | SUPPORTED | Step 7 Verification (Drift recognized with degraded health) |
| Frozen/Stuck Sensor | SUPPORTED | Step 9 E2E (Flatline perfectly detected) |
| Offset | SUPPORTED | Step 9.2 (Dynamic Baseline successfully detected offset) |
| Noise | NOT VALIDATED / LIMITATION | Step 9.2 (Mild noise evaded smoothing filters natively) |
| Missing/Data Loss | SUPPORTED | Step 9.1 (API successfully routed `null` payload to `DATA_LOSS`) |
| Multivariate Inconsistency | SUPPORTED | Step 9.2 (Dynamic baseline offset detected as structural disagreement) |
| Unknown | SUPPORTED | Step 6 V2 natively outputs `UNKNOWN` fallback hint if thresholds met without strong categorical match. |

## 3. SEVERITY
- **Audit**: Severity (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`) is deterministically mapped directly from the `final_anomaly_score` bins [0.0 - 1.0]. It is output via WebSocket and visualizable on the dashboard.
- **Status**: SUPPORTED.

## 4. CONFIDENCE
- **Audit**: Confidence is produced distinctly from `anomaly_score`. It evaluates the structural agreement of multiple evidence vectors (e.g., if Temporal, Statistical, and Multivariate all agree on a Sudden Event, Confidence approaches 1.0).
- **Status**: SUPPORTED. (Note: Confidence is an uncalibrated heuristic score representing structural evidence alignment, NOT a statistically calibrated probability curve).

## 5. ROOT-CAUSE / FAULT CLASSIFICATION
- **Audit**: The model natively outputs a `fault_type_hint` containing deterministic categorical explanations based strictly on the highest mathematically triggered evidence family.
- **Status**: SUPPORTED. (Note: Claiming "100% perfect classification accuracy" must be avoided).

## 6. TEMPORAL PATTERNS
- **Audit**: The `StreamPipeline` retains a rolling 360-minute in-memory window. It computes explicit causal temporal derivatives: `sampling_interval_mins`, `delta_1`, `rate_per_hour`, `roll_var_60m`, and persistence streaks (`consec_unchanged`).
- **Status**: SUPPORTED.

## 7. MULTIVARIATE CONSISTENCY
- **Audit**: Temperature, Pressure, and Humidity are analyzed structurally. The pipeline explicitly calculates 60m z-score disagreements (`multivariate_z_disagreement`) and dominant structural divergence variables.
- **Status**: SUPPORTED. (Note: Do not claim full physical atmospheric or thermodynamic modeling).

## 8. EXPLAINABILITY
- **Audit**: The API exports a clear English `explanation` string mapping directly to the underlying activated logic gate. No LLMs are utilized, ensuring strict mathematical determinism.
- **Status**: SUPPORTED.

## 9. SENSOR HEALTH
- **Audit**: Step 7 `SensorHealthTracker` implements a strictly bounded [0,100] state engine mapping to `HEALTHY`, `WATCH`, `DEGRADED`, and `CRITICAL`. Persistence penalizes health exponentially; recovery occurs safely over sequential normal observations.
- **Status**: SUPPORTED.

## 10. DATA QUALITY / COMMUNICATION ERRORS
- **Audit**: `all_sensors_missing` sets `data_quality_status = DATA_LOSS`. Partial nulls set `DEGRADED`. The step 7 engine intentionally separates communication failures from hardware physical decay.
- **Status**: SUPPORTED.

## 11. OPTIONAL CORRECTED / IMPUTED VALUES
- **Audit**: The pipeline internally imputes `NaN` via medians specifically to pass `sklearn` inference without crashing, but it does NOT explicitly correct or emit a "clean" observation back to the user payload as a distinct time-series.
- **Status**: OPTIONAL / NOT IMPLEMENTED.

## 12. SPATIAL / NEIGHBORING-STATION CONSISTENCY
- **Audit**: The current engine strictly processes one station pipeline in complete isolation.
- **Status**: NOT IMPLEMENTED. (Note: Multivariate T/P/RH consistency is not neighboring-station spatial interpolation).

## 13. DASHBOARD / VISUALIZATION
- **Audit**: The React dashboard displays live metrics, fault types, sensor health gauges, event timelines, and active anomalies fetched via WebSocket.
- **Status**: SUPPORTED. (Note: Browser E2E automated validation was not performed).

## 14. SCALABILITY
- **Audit**: Currently scales vertically using isolated in-memory Python dictionaries keyed by `station_id`.
- **Status**: PARTIAL. (Distributed deployment across multiple load-balanced Kubernetes pods would require migrating this state to an external cache like Redis).

## 15. DEPLOYABILITY
- **Audit**: The repository contains instructions to run FastAPI and the frontend stack locally.
- **Status**: SUPPORTED. (Local). (Note: CI/CD cloud production pipelines do not exist).

## 16. ENERGY EFFICIENCY
- **Audit**: No hardware/power benchmark was performed on the software prototype.
- **Status**: NOT MEASURED.

## 17. ACCURACY
- **Row-Level Precision**: 60.14% (V2)
- **Row-Level Recall**: 37.05% (V2)
- **F1 Score**: 0.4585 (V2)
- **Event-Level Recall**: 90.77%
- **False Positives**: 5,169 out of 2.5 million rows (V2)
- **Context**: The strict unsupervised percentile-threshold strategy suppresses noisy FPs beautifully but creates a mathematical limitation for row-level recall on highly subtle variations. Event-level accuracy (90.7%) is the primary viable operational metric.

---

## CLAIM AUDIT TABLE

| Claim | Evidence | Safe to claim? | Correct wording |
|---|---|---|---|
| "The system is 100% accurate." | Event recall is 90.7%; FP exist. | NO | "The system achieved 90.7% event-level recall on the synthetic evaluation dataset." |
| "It operates instantaneously." | Pipeline latency is ~80ms. | NO | "It operates in real-time with sub-100ms backend processing latency." |
| "Perfectly predicts sensor noise." | Mild noise evaded detection. | NO | "Structurally isolates large shifts; mild noise organically evades thresholds by design." |
| "Fully scalable architecture." | In-memory `dict` state. | NO | "Station logic is vertically scalable and isolated. Distributed deployment would require an external/shared state strategy." |
| "Zero False Positives." | 5,169 FPs triggered in validation. | NO | "Maintained a low false-positive footprint through multivariate context suppression." |
| "Validates missing-data limits." | JSON null validated successfully. | YES | "Missing payloads natively trigger `DATA_LOSS` logic independent of sensor hardware degradation." |
