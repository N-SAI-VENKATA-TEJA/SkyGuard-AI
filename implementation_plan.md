# Step 7: Fault Classification + Sensor Health Engine Implementation Plan

## Goal Description
Implement Step 7 to consume Step 6 V2 outputs and produce operational, stateful answers about the current state of the AWS station. The engine performs fault classification, sensor attribution, and dynamic health tracking. The process remains strictly unsupervised, deterministic, causal, and operates without LLMs or heavy ML models.

## Open Questions
None. The specifications for penalties, recovery, time gaps, sensor attribution, and required testing have been exhaustively provided.

## Proposed Changes

### Core Logic `ml/anomaly_engine/step7_sensor_health.py`
[NEW] `ml/anomaly_engine/step7_sensor_health.py`
Contains the `SensorHealthTracker` class, designed to process observations iteratively and strictly causally.

- **Persistence Tracking & Natural Time Gaps**:
  - Time-gap aware streak counters (`anomaly_streak`, `sensor_fault_streak`, `missing_streak`, `frozen_streak`).
  - A timestamp gap indicates missing observations. It does *not* indicate recovery or `DATA_LOSS_COMMUNICATION` automatically.
  - If gap > continuity tolerance (e.g., 30 mins), persistence counters are **reset**.
  - No health penalties or recoveries are applied merely because of the time gap.

- **Fault Classification**:
  - Valid types: `NORMAL`, `SPIKE`, `DRIFT`, `FROZEN`, `OFFSET`, `NOISE`, `MISSING`, `MULTIVARIATE_INCONSISTENCY`, `UNKNOWN`.
  - Driven by Step 6 `fault_type_hint` combined with accumulated evidence and persistence.

- **Sensor Attribution**:
  - Computes a sensor-specific evidence strength using temporal, statistical, stability, deviation, missingness, and Step 6 evidence for TEMPERATURE, PRESSURE, and HUMIDITY.
  - If one sensor is dominant $\to$ `affected_sensor = that_sensor`.
  - If multiple sensors are genuinely implicated $\to$ `MULTIPLE`.
  - If all sensors missing $\to$ `ALL_SENSORS`.
  - If evidence is insufficient $\to$ `UNKNOWN`. Do not force attribution.

- **Data Quality vs. Health Score**:
  - `data_quality_status`: `GOOD`, `DEGRADED`, `DATA_LOSS`, `UNKNOWN`.
  - Communication losses (missing data) result in `DATA_LOSS` but do **not** automatically penalize physical sensor health scores.

- **Health Engine**: 
  - Stateful tracker for each sensor, strictly bounded to `[0, 100]`.
  - **Penalty Formula**: 
    $penalty = base\_fault\_penalty \times evidence\_strength \times confidence \times persistence\_factor$
    where $persistence\_factor = \min(MAX\_PERSISTENCE, 1 + growth \times fault\_streak)$.
    This guarantees bounded, progressive degradation for long streaks without creating arbitrary/unexplained numbers.
  - **Recovery Formula**: 
    Gradual bounded recovery awarded only on healthy observations (no relevant fault evidence, no active persistence condition). Does not instantly return a degraded sensor to 100.
  - **Health Update**: $health_t = \text{clip}(health_{t-1} + recovery - penalty, 0, 100)$

- **Status Thresholds**:
  - Explicitly documented as *operational interpretation thresholds*:
    - $\ge 90 \to$ `HEALTHY`
    - $\ge 70 \to$ `WATCH`
    - $\ge 40 \to$ `DEGRADED`
    - $< 40 \to$ `CRITICAL`

- **Maintenance Recommendations & Explanations**: 
  - Deterministic operational recommendations (e.g., "Inspect sensor calibration due to persistent offset/drift pattern.").
  - Clear explanations citing the affected sensor, relevant evidence, persistence, and confidence.

### Pipeline Runner `ml/anomaly_engine/run_step7.py`
[NEW] `ml/anomaly_engine/run_step7.py`
- Loads `data/processed/hybrid_predictions_v2.csv` and necessary evidence components.
- Initializes `SensorHealthTracker` and iterates chronologically.
- Exports to `data/processed/step7_sensor_health.csv`.
- Benchmarks processing time and rows/sec for real-time viability.

### Validation & Testing
[NEW] `ml/anomaly_engine/evaluate_step7.py`
- Evaluates overall distributions and verifies the distinction between data quality drops and health score drops.

[NEW] `tests/test_step7_sensor_health.py`
- Deterministic synthetic scenario tests:
  1. Isolated temperature spike
  2. Repeated temperature spikes
  3. Temperature frozen for long period
  4. Temperature drift
  5. Pressure offset
  6. Humidity noise
  7. All sensors missing
  8. Partial sensor missing
  9. Multivariate inconsistency
  10. Long healthy period after anomaly
  11. **[NEW]** `HEALTHY \to ANOMALY \to NATURAL TIME GAP \to HEALTHY` to verify penalty application, streak reset across gap, lack of recovery during gap, and gradual post-gap recovery.
- **Natural Time Gaps**: Explicitly test the major dataset gap (2016-10-25 10:30 $\to$ 2016-10-28 12:50).
- **Mandated Causality Test**: Verifies $state_t$ remains unchanged when future data $t+10$ is appended.

### Documentation
[NEW] `docs/step7_sensor_health.md`
- Documentation of formulas, thresholds, attribution logic, recovery behavior, and limitations.

[NEW] `docs/step7_verification.md`
- Reporting scenario tests, gap handling, causality validation, performance benchmarks, and checklist.

## Verification Plan
1. **Automated Tests**: Run `pytest tests/test_step7_sensor_health.py` to assert all 11 scenarios, causality checks, and gap handling logic exactly match specifications.
2. **Evaluation Metrics**: Run `python ml/anomaly_engine/evaluate_step7.py`.
3. **Manual Verification**: Review CSV outputs to ensure communication failures affect data quality but not sensor hardware health, and that time gaps successfully reset streaks.
