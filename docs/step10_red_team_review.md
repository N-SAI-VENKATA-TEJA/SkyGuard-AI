# STEP 10: RED-TEAM REVIEW

## 1. What would an evaluator challenge?
Evaluators will likely challenge the lack of neighboring-station spatial validation, the lack of cloud deployment demonstrations, and the decision to construct a contaminated (5% synthetic anomaly) target dataset rather than sourcing labeled historic meteorological disasters.

## 2. What is our strongest technical feature?
The deterministic `SensorHealthTracker` (Step 7) tied directly to the Hybrid Contextual Suppression (Step 6). By intelligently rejecting multivariate weather shifts, isolating true physical degradation, and persistently scoring health iteratively independent of data loss, the solution elevates from a theoretical math model into an actionable operational product.

## 3. What is our weakest technical area?
Subtle noise evasion and the dependence on an in-memory `dict` state in `StreamPipeline` for station history.

## 4. What experiment would expose a weakness?
Restarting the FastAPI server midway through an ongoing drift anomaly. Because the pipeline is in-memory and volatile, the server reboot would annihilate the 360-minute historical context. The subsequent observations would enter "WARMUP" state, completely failing to detect the ongoing drift until 36 observations reconstruct the baseline.

## 5. Which claims could be attacked?
Any claim of "High Accuracy" based on synthetic evaluations. A judge would argue that synthetic anomalies injected programmatically do not perfectly mimic real meteorological failures, limiting confidence in real-world deployment accuracy.

## 6. Which parts are genuinely innovative?
The `EVIDENCE_AGREEMENT` structural logic in `HybridAnomalyEngineV2`. Instead of relying entirely on `sklearn` output probabilities, the system explicitly intersects Temporal, Statistical, and Multivariate structural evidence independent of the models themselves to formulate a deterministically explainable fallback.

## 7. Which parts are standard ML engineering?
The `Statistical Baseline`, `PCA`, and `Isolation Forest` ensembles. The baseline predictors run off standard implementations and traditional rolling temporal features.

## 8. What questions could judges ask about PCA?
*Why did PCA produce so many False Positives?*
**Answer:** PCA mathematically flags variance along minor principal components. Natural weather transitions inherently distribute variance unexpectedly across these components, causing standard PCA reconstruction error thresholds to spike during safe, natural atmospheric changes.

## 9. Why hybrid detection?
Because no single mathematical definition of "anomaly" is correct. An Isolation Forest is exceptional at finding density outliers (Spikes) but struggles tracking highly dense gradual calibration drifts (Drift). Standard thresholds fail. A Hybrid system intersections the models to leverage their varying strengths.

## 10. Why not deep learning?
Deep Learning (LSTMs, Transformers) requires massive, perfectly-labeled sequences for supervised target-learning, which do not exist here. Furthermore, they act as black-boxes. Operational meteorologists require strict deterministic explainability (e.g., "Temperature rate-of-change breached multivariate bounds"), which our contextual hybrid framework delivers.

## 11. Why synthetic anomalies?
Because acquiring a unified, strictly-labeled dataset of every type of meteorological hardware failure (Stuck, Drift, Missing, Spike) spanning years of normal background variance is nearly impossible in the public domain. Synthetic injection allows us to establish verifiable ground-truth bounding boxes to evaluate standard algorithms.

## 12. How would this work with real AWS data?
The existing pipeline has successfully processed `aws_clean.csv` (a real 450,000+ row historic sequence) cleanly in Step 9.2. It accurately maps the natural variances without hallucinating continuous warnings, proving real-world compatibility.

## 13. How would multiple stations be handled?
Currently handled securely via dynamic dictionary instantiation where `station_id` keys map to completely isolated memory pipelines. 

## 14. What happens during communication loss?
Null JSON fields trigger `pd.isna()`. This completely isolates the `DATA_LOSS` logic, returning immediate critical UI alerts without mistakenly degrading the physical sensor hardware health score.

## 15. How is confidence different from score?
Score is the raw maximum mathematically intersected threshold breach (Severity). Confidence measures how many distinct feature domains (Temporal + Statistical + Multivariate) independently agreed on the score. 

## 16. How is sensor health different from anomaly detection?
Anomaly detection is point-in-time contextual event discovery. Sensor Health is a persistent, exponentially decaying memory of anomalies over time. A single anomaly creates a blip; a persistent anomaly crashes the health gauge and triggers maintenance action.

## 17. What happens during extreme but genuine weather?
Contextual Suppression protects the system. If Temperature spikes, but Pressure and Humidity shift coherently alongside it in natural physical alignment, the `multivariate_z_disagreement` remains low, and the system suppresses the anomaly score artificially back into `NORMAL`.

## 18. Can the system distinguish a genuine extreme event from a sensor fault with certainty?
No system can with 100% certainty without spatial neighboring-station validation. However, multivariate consistency analysis allows the system to suppress the vast majority of genuine extreme weather.

## 19. What are the current limitations?
- Stateful memory is volatile.
- Lacks spatial cross-station validations.
- Imputation (while internal to sklearn) is not broadcasted as a corrected feed.
- Small noise anomalies easily evade structural boundaries.
