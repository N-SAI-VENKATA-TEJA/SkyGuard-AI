# Step 6 V2: Threshold & Saturation Audit

## 1. Investigation Objective
The reported V2 development threshold of `1.000` for both Sudden and Persistent fault families was flagged for investigation. We conducted a deep audit to determine whether this is a legitimate consequence of the score distributions, a symptom of structural saturation, or a byproduct of evaluation label leakage.

## 2. Pre-Clip Score Distributions

The V2 architecture applies a mathematical clip: `final_score = np.clip(core + bonus + capped_model, 0.0, 1.0)`. To understand the `1.000` threshold, we audited the raw scores *before* clipping on the Development dataset.

### Sudden Event Pre-Clip Scores
| Subset | Median | P90 | P95 | P99 | P99.5 | Max | Fraction $\ge$ 1.0 |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.245 | 0.634 | 0.764 | 1.061 | 1.198 | 1.500 | **1.26%** |
| **Anomalous** | 0.305 | 0.735 | 0.947 | 1.400 | 1.498 | 1.500 | **3.96%** |

### Persistent Fault Pre-Clip Scores
| Subset | Median | P90 | P95 | P99 | P99.5 | Max | Fraction $\ge$ 1.0 |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.115 | 0.217 | 0.346 | 0.527 | 0.674 | 1.111 | **0.03%** |
| **Anomalous** | 0.200 | 1.168 | 1.259 | 1.300 | 1.300 | 1.300 | **15.11%** |

## 3. Findings: Why the 99.5th Percentile is exactly 1.000

The `1.000` development thresholds are structurally inevitable due to two distinct phenomena, neither of which are bugs, but represent fundamental challenges in unsupervised scoring:

1. **Sudden Event Saturation (The Clipping Effect):** 
   Even after adjusting the exponential bounding constants (`k`) to accommodate high natural weather variance, approximately **1.26%** of legitimate normal rows naturally experience extreme multivariate transitions that push `core + bonus + model_support` slightly above `1.0`. Because the score is clipped at 1.0, these rows form a mass at exactly `1.000`. Taking the 99.5th percentile of normal rows lands squarely inside this 1.26% mass, pinning the threshold to `1.000`.
   
2. **Persistent Fault Contamination (The Outlier Masking Effect):** 
   For Persistent faults, normal rows behave beautifully (99.5th percentile is only `0.674`). However, the development dataset contains ~5% injected anomalies. Since the 15% most severe anomalies legitimately max out at 1.000, they easily dominate the top 0.5% of the entire development dataset. Thus, when we take the 99.5th percentile of the *entire contaminated dataset*, we are sampling the anomaly cluster itself, returning a threshold of `1.000`.

*Conclusion: The thresholds were mathematically accurate based on the unsupervised strategy, but represent a saturation barrier. Family-specific thresholds are indeed calculated independently, but both converged to 1.000 for these structural reasons.*

## 4. Re-Evaluation Post-Implementation Correction
To mitigate the saturation observed in Sudden Events, we widened the exponential bounding constants ($k=25, 30, 30$) to further accommodate extreme natural weather variance. 

### Final Metrics (Evaluation Dataset)
- **Evaluated Rows:** 420,224
- **Total True Anomaly Rows:** 21,051
- **Row Precision:** 60.14%
- **Row Recall:** 37.05%
- **Row F1 Score:** 45.85%
- **False Positives:** 5,169 rows (Down from 133,000+ in PCA)
- **False Negatives:** 13,252 rows
- **Event-Level Recall:** 90.77%

### Verification Checks
- **Causality Check:** **PASS**. Iteratively appending data $t+10$ did not alter scores computed at time $t$. All baselines are strictly past-only.
- **Contextual Independence:** **PASS**. Persistent sensor faults bypassed the meteorological suppression logic.
- **Communication Identification:** **PASS**. Missing events achieved a perfect 1.0 communication score independently.

## 5. Summary
The engine operates safely and causally. The 1.0 threshold is a major contributor to reduced row-level recall, particularly for subtle anomalies whose final scores remain below the strict alert boundary.

**Event-Level Recall Definition:**
- **Anomaly Event:** A distinct incident defined by a unique `anomaly_id` in the ground-truth labels.
- **Detected Event:** An anomaly event where the maximum `anomaly_flag` across its duration is `True`.
- **Matching Rule:** One overlapping predicted row during the injected event duration is sufficient to count as a detected event.

90.77% event-level recall means that 90.77% of defined anomaly events had at least one qualifying detection under the documented event-matching rule.

Under the specified unsupervised percentile-thresholding strategy, the contaminated development distribution produces a strict 1.000 threshold.
