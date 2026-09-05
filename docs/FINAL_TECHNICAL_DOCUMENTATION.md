# SIH26073: SkyGuard AI Final Technical Documentation

## 1. System Architecture
The SkyGuard AI system is a real-time, streaming pipeline designed for the detection of telemetry faults in Automatic Weather Stations. 
The system operates sequentially:
1. **API Gateway:** A FastAPI application intercepts live 10-minute frequency payloads containing Temperature, Pressure, and Humidity.
2. **Stateful Streaming:** The observation is forwarded to a station-specific `StreamPipeline` instance that buffers the last 360 minutes of causal history in volatile memory.
3. **Causal Feature Engineering:** 89 causal, look-ahead-free features (temporal rates, rolling variance, multivariate divergences) are extracted from the memory buffer.
4. **Frozen ML Detection:** The feature vector passes through Statistical, PCA, and Isolation Forest frozen baselines.
5. **Hybrid V2 Engine:** Predictions are evaluated contextually, suppressing natural weather transitions and generating an aggregated anomaly score.
6. **Sensor Health:** A continuous state engine degrades physical health if faults persist and awards recovery during sustained normal operation.
7. **Broadcast:** Results are emitted via WebSocket to a React dashboard.

## 2. Data Pipeline
The core data source is the Max Planck Weather Dataset. The system utilizes `simulator/generate_dataset.py` to synthesize specific sensor faults onto the chronologically sorted baseline. This synthetic strategy provides deterministic ground-truth labels for evaluation.

## 3. Feature Engineering
The pipeline executes within `ml/features/stream_feature_engineer.py`. 
Key structural domains include:
- **Temporal Rates:** First-order and rate-per-hour derivatives.
- **Statistical Envelopes:** 60-minute rolling medians, variances, and IQRs.
- **Persistence Tracking:** Sequential unchanged readings (e.g., `consec_unchanged`).
- **Multivariate Disagreement:** Z-score divergences explicitly comparing T, P, and RH structures.
- **Missing Flags:** Explicit detection of nan/null representations for data loss handling.

## 4. Detector Architecture & Hybrid V2
Instead of routing raw observations to a black-box deep learning model, the architecture relies on interpretable intersections:
- **Isolation Forest:** Optimized for density variations (Spikes).
- **PCA:** Analyzes reconstruction loss across combined variables.
- **Statistical Baseline:** Enforces hard thermodynamic operational bounds.

**Hybrid V2 Logic:**
The baseline models generate raw predictions. The Hybrid V2 engine `ml/anomaly_engine/hybrid_detector_v2.py` aggregates these scores and then mathematically gates them using Contextual Suppression. If the statistical bounds remain unbroken and the multivariate variables move coherently, the anomaly score is suppressed. The system only broadcasts an alert when the evidence supports a structural hardware fault over a natural atmospheric event.

## 5. Sensor Health
The `SensorHealthTracker` implements a bounded [0,100] state. Unlike the anomaly flag (which is a point-in-time event marker for a single observation), the health score incorporates persistent memory across observations. A single anomaly creates a minor penalty. Persistent streaks create exponential penalties. Validating missing data triggers `DATA_LOSS` but safely bypasses the hardware physical health decay algorithm.

## 6. Streaming & API
The backend operates entirely in memory using a dictionary keyed by `station_id`. This isolation allows multiple weather stations to run concurrently on a single process without data leakage. The `FastAPI` endpoint integrates with `pydantic` schemas, validating types and propagating safely. `WebSocket` managers push immediate broadcast strings formatted for the UI.

## 7. Dashboard
The React Dashboard connects asynchronously. It charts historical sliding windows of T/P/RH alongside corresponding anomaly markers. The dashboard dynamically updates localized views of sensor health, explicit AI-generated explanations, and multi-station selector streams.

## 8. Evaluation Methodology & Metrics
The pipeline was evaluated against a synthetic 5% fault-injected evaluation set. These metrics were obtained on the project's synthetic evaluation dataset using the defined evaluation methodology and do not represent general real-world accuracy.
- **Event-Level Recall:** 90.77% — the fraction of distinct injected anomaly events where at least one observation was correctly flagged.
- **Row-Level Precision:** 60.14% — greatly enhanced by Hybrid contextual suppression vs PCA baseline (6.7%).
- **Row-Level Recall:** 37.05% — reflects the strict unsupervised threshold strategy.
- **StreamPipeline Latency:** ~47.11 ms/observation (measured local benchmark, not a guaranteed production value).
- **API Benchmark Mean:** ~83.36 ms/observation (includes HTTP overhead; max latency ~1640 ms on first-call cold-start due to Joblib artifact loading).

## 9. Limitations
Detailed in `docs/TECHNICAL_LIMITATIONS.md`. Key considerations include volatile in-memory state tracking, lack of spatial verification between physically neighboring AWS stations, and threshold evasion by low-variance noise.
