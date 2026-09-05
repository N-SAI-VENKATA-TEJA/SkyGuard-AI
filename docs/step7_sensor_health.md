# Step 7: Fault Classification & Sensor Health Engine

## 1. Purpose
The Step 7 engine translates the continuous, raw anomaly scores from Step 6 into operational language. It answers "What kind of problem does this observation represent, which sensor is involved, and what is the current health state of that sensor?" The engine is strictly deterministic, stateful, and unsupervised (no LLMs or heavy ML models).

## 2. Input
The engine consumes:
- Step 6 outputs: `anomaly_flag`, `candidate_fault_family`, `fault_type_hint`, `final_anomaly_score`, `anomaly_confidence`.
- Feature Engineering components: Temporal rates (`rate_per_hour`) and statistical deviations (`roll_z_60m`).
- Original sensor values (`temperature`, `pressure`, `humidity`).

## 3. Fault Classification
The engine classifies anomalous events into explicit types:
- `NORMAL`, `SPIKE`, `DRIFT`, `FROZEN`, `OFFSET`, `NOISE`, `MISSING`, `MULTIVARIATE_INCONSISTENCY`, `UNKNOWN`.
Classification is driven by Step 6's hints in conjunction with local missingness indicators.

## 4. Sensor Attribution
Sensor attribution identifies exactly which hardware is degrading. 
- Evidence is gathered per-sensor using both normalized temporal rates and statistical Z-scores.
- If a sensor's evidence dominates strongly, it is attributed to that sensor (`TEMPERATURE`, `PRESSURE`, `HUMIDITY`).
- If multiple sensors exhibit strong, indistinguishable failure patterns, it maps to `MULTIPLE`.
- `DATA_LOSS_COMMUNICATION` maps to `ALL_SENSORS`.
- If evidence is ambiguous, the engine safely falls back to `UNKNOWN` to avoid false attribution.

## 5. Health Score Formula
Each sensor maintains a `health_score` $\in [0, 100]$. The formula is applied at each timestep iteratively:
$$ health_t = \max(0, \min(100, health_{t-1} + recovery - penalty)) $$

**Penalty Formula:**
$$ penalty = base\_penalty \times evidence \times confidence \times persistence\_factor $$
Where:
- $base\_penalty$: Fault-specific constant (e.g., Spike: 5.0, Frozen: 5.0, Missing: 0.0).
- $evidence$: Normalized [0,1] confidence based on physical deviation.
- $persistence\_factor = \min(5.0, 1.0 + 0.5 \times fault\_streak)$ 

This guarantees bounded, progressive degradation for long streaks without causing an instant collapse on an isolated anomaly.

## 6. Health Recovery
Recovery is bounded and gradual. A sensor receives $+0.2$ health points for every normal observation where no relevant fault evidence or active persistence condition is detected. A degraded sensor requires continuous healthy operation over multiple hours to return to 100%.

## 7. Persistence Tracking & Time Gaps
The engine maintains streak counters (`anomaly_streak`, `frozen_streak`, `missing_streak`, `sensor_fault_streak`). 
**Time Gap Logic:**
- If a timestamp gap $> 30$ minutes occurs, the engine assumes missing continuity and **resets** persistence counters.
- Time gaps do NOT independently cause health penalties or grant health recoveries.
- A time gap is handled as a period of no information, not equivalent to a known communication outage.

## 8. Data Quality vs Sensor Health
A core principle of Step 7 is separating telemetry issues from hardware failures.
- **Data Quality Status**: `GOOD`, `DEGRADED` (partial loss), or `DATA_LOSS` (full loss).
- If all sensors are missing, `data_quality_status = DATA_LOSS`, but the physical `health_score` of the sensors is NOT penalized (base penalty = 0.0).

## 9. Maintenance Status (Operational Recommendations)
Based on streaks and current classification, the engine recommends operational actions:
- `NO_ACTION`: Healthy operation.
- `MONITOR`: Short-term anomalies (e.g., brief isolated spikes).
- `INSPECT`: Persistent drift or repeated anomalies on the same sensor.
- `MAINTENANCE_RECOMMENDED`: Severe persistent issues (e.g., heavily frozen sensors).
- "Check AWS communication/telemetry connection.": Triggered by missing data events.

*Note: This is an operational recommendation, not a predictive failure model.*

## 10. Causality
The entire engine is stateful and purely causal. Step $t$ computes solely based on the state at $t-1$ and observations $\le t$. 

## 11. Limitations
- Unsupervised logic means that extremely long stretches of legitimate natural variance, if incorrectly flagged by Step 6 due to 1.0 threshold saturation, will cause transient, false health degradation.
- Sensor attribution relies on basic heuristic combinations of Z-score and temporal derivatives; complex cross-sensor corruptions may resolve to `MULTIPLE` or `UNKNOWN`.
