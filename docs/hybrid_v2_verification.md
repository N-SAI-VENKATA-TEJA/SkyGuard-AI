# Hybrid Engine V2 Verification

## 1. Metrics Improvement over PCA
- **Precision**: Increased from 0.067 (PCA) to 0.6014 (V2).
- **Row-Level Recall**: 0.3705 (V2) compared to 0.460 (PCA).
- **F1 Score**: Increased from 0.118 (PCA) to 0.4585 (V2).
- **False Positives (FP)**: 5,169 (V2) compared to 133,108 (PCA).
- **False Negatives (FN)**: 13,252 (V2).
- **Event-Level Recall**: 90.77%.

**Event-Level Recall Definition:**
- **Anomaly Event:** A distinct incident defined by a unique `anomaly_id` in the ground-truth labels.
- **Detected Event:** An anomaly event where the maximum `anomaly_flag` across its duration is `True`.
- **Matching Rule:** One overlapping predicted row during the injected event duration is sufficient to count as a detected event.

90.77% event-level recall means that 90.77% of defined anomaly events had at least one qualifying detection under the documented event-matching rule.

## 2. False Positive Reduction
- Total False Positives dropped from 133,108 (PCA) to 5,169 (V2).
- By capping model support at 0.20 and implementing Contextual Suppression, the majority of legitimate weather transitions were correctly ignored.

## 3. Architecture Validation
- **Causality Test:** PASS. Iteratively appending data $t+10$ did not alter scores computed at time $t$. 
- **Contextual Independence:** PASS. Persistent sensor faults (e.g., frozen sensors) correctly bypassed meteorological suppression logic.
- **Communication Identification:** PASS. Missing events achieved a perfect 1.0 communication score independent of statistical models.

## 4. Known Limitations
Because the development dataset contains ~5% injected synthetic anomalies, and the V2 architecture properly isolates fault evidence to allow clear 1.0 scoring, the 99.5th percentile threshold on the development set is mathematically drawn directly from the anomaly population (1.000). The 1.0 threshold is a major contributor to reduced row-level recall, particularly for subtle anomalies whose final scores remain below the strict alert boundary. 

Under the specified unsupervised percentile-thresholding strategy, the contaminated development distribution produces a strict 1.000 threshold.
