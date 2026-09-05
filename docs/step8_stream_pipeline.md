# Step 8.2: Stream Pipeline

## 1. Architecture
The `StreamPipeline` is a strictly causal, stateful wrapper around the frozen intelligence of Step 6 V2 and Step 7. It allows processing a single incoming observation `(timestamp, temperature, pressure, humidity)` without needing access to the full dataset history.

```ascii
[New AWS Observation]
         |
[StreamFeatureEngineer]  <-- Maintains 360m Deque + Streak Counters
         |
[Step 5 Models (Frozen)] <-- Generates Model Evidence Scores
         |
[Step 6 V2 Engine (Frozen)] <-- Fuses Evidences & Categorizes Anomaly
         |
[Step 7 HealthTracker]   <-- Updates Physical Health state
         |
[Operational Result dict]
```

## 2. State Management
The streaming feature engineering maintains bounded state to support Step 4 features:
- **history deque**: Maximum 360 minutes of raw observations to support `_roll_mean_360m`, etc.
- **consec_unchanged**: Counters storing stability streaks.
- **last_obs**: The immediate prior observation $t-1$ for computing first-order derivatives.

## 3. Causal Feature Generation
All rolling features are calculated using a strict `closed='left'` emulation. For any observation at time $t$, the rolling statistics are computed strictly on observations $t_i < t$. The observation at $t$ is appended to the deque *after* feature generation. This guarantees no future information leaks into the present detection.

## 4. Warm-Up
A "warm-up" period is required to fill the deque before rolling statistical deviations (`roll_z`) can be securely calculated. Since the maximum historical window is 360 minutes, the pipeline explicitly requires 36 observations (assuming standard 10-minute intervals). We define the WARMUP state for the first 35 observations. During WARMUP:
- Step 6 logic continues but the pipeline forcibly marks `processing_state = 'WARMUP'` and `anomaly_flag = False`.
- Step 7 receives the suppressed anomaly, preventing artificial health penalties from algorithmic cold starts.

## 5. Time Gaps and Missing Data
- **Gaps**: A timestamp gap $> 30$ minutes does NOT penalize health and does NOT award recovery. It safely resets Step 7 persistence streaks as per the frozen logic. The 360-minute deque naturally purges expired rows, resuming calculations cleanly on post-gap observations.
- **Missing Data**: Explicitly missing sensors (`NaN`) are preserved through the pipeline, triggering `DATA_LOSS` natively.

## 6. Output Contract
Each processed row returns a dictionary containing:
- Raw inputs
- `anomaly_score`, `anomaly_flag`, `severity`, `confidence`
- `fault_type`, `affected_sensor`
- Continual `health` $[0, 100]$ and `maintenance_status`
- Textual recommendations
