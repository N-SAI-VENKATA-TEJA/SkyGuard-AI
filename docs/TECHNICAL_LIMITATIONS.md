# SkyGuard AI: Technical Limitations

## 1. Mild Noise Evasion
- **Limitation:** The pipeline frequently fails to detect minor interference/noise anomalies (e.g., standard deviation of 1.5).
- **Why it exists:** The rolling statistical boundaries are mathematically designed to absorb natural atmospheric volatility. Tightening these bounds to catch minor noise would result in an unacceptable flood of false positives during genuine weather shifts.
- **Impact:** Low; mild noise rarely affects the operational integrity of weather forecasts.
- **Future Improvement:** Implementation of targeted high-frequency spectral analysis (Fourier transforms) specifically tuned for electronic interference, separate from the macroscopic meteorological bounds.

## 2. In-Memory State Volatility
- **Limitation:** The `StreamPipeline` stores the 360-minute causal history internally as a Python dictionary.
- **Why it exists:** Rapid prototyping and zero-dependency local deployments.
- **Impact:** If the FastAPI backend pod crashes or restarts, all active historical baseline context is erased. The pipeline must endure a 360-minute `WARMUP` phase before inference can resume.
- **Future Improvement:** Externalize state management to a persistent KV store (e.g., Redis).

## 3. Lack of Spatial (Neighboring-Station) Validation
- **Limitation:** A station is evaluated entirely in isolation.
- **Why it exists:** The project scope was constrained to single-station multivariate telemetry without geographic metadata integration.
- **Impact:** Extreme micro-climate weather events could theoretically trick the system into flagging a fault. Cross-referencing a station 5km away would immediately resolve this ambiguity.
- **Future Improvement:** Implement a spatial aggregator node that compares Z-scores across geofenced AWS clusters.

## 4. Synthetic Ground Truth Dependency
- **Limitation:** All accuracy metrics (90.77% Event Recall, 60.14% Row Precision) were evaluated on a synthetically contaminated historical dataset.
- **Why it exists:** A comprehensive, publicly available dataset documenting every permutation of AWS hardware failure with millisecond-accurate timestamps does not exist.
- **Impact:** The models are highly optimized to detect our specific mathematical formulations of faults. Real-world physical decay may present novel, undetected degradation curves.
- **Future Improvement:** Partnering with a meteorological agency to acquire genuine maintenance logs mapped to historic telemetry for fine-tuning.

## 5. Non-Broadcasted Imputation
- **Limitation:** While the pipeline internally interpolates/imputes missing values to satisfy `sklearn` array constraints, it does not broadcast a parallel "cleaned/imputed" time-series back to the user.
- **Why it exists:** The focus was strictly anomaly detection and sensor health tracking, not data reconstruction.
- **Impact:** Users are alerted to the fault, but their downstream applications do not receive an automatically repaired data feed.
- **Future Improvement:** Output an `imputed_value` dictionary within the JSON payload.
