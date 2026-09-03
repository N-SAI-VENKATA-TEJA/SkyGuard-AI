# Step 6 V2: Fault-Family Hybrid Anomaly Engine

## 1. Architectural Philosophy
The Step 6 V2 engine corrects the "structural dilution" failure of the V1 engine. Instead of a global weighted average that forced independent faults to compensate for a lack of unrelated evidence, V2 utilizes independent **Fault Families**:
1. `SUDDEN_EVENT`: Spikes, offsets, noise, and sudden multivariate inconsistencies.
2. `PERSISTENT_SENSOR_FAULT`: Frozen sensors and sustained drift.
3. `DATA_LOSS_COMMUNICATION`: Missing data events.

By isolating fault evidence, the system allows a pure "frozen" anomaly to assert 1.0 evidence within its family without being diluted by the absence of "spike" evidence. 

## 2. Evidence Fusion Formulas
Each continuous fault family (Sudden and Persistent) evaluates its relevant evidence layers (bounded via $1 - e^{-x/k}$).
The family fusion is given by:
`family_score = min(strongest_core_evidence + agreement_bonus + capped_model_support, 1.0)`
- `strongest_core_evidence`: The maximum of the family's primary evidence layers.
- `agreement_bonus`: $+0.10$ for each independent core evidence layer $\ge 0.30$.
- `capped_model_support`: Step 5 normalized model scores, strictly capped at $0.20$ to prevent PCA false-positive domination.

## 3. Contextual Suppression
Contextual suppression is applied **only** to the `SUDDEN_EVENT` family.
If a sensor exhibits high deviation but `multivariate_z_disagreement` is below 2.0 (indicating coherent multivariate movement), a `CONTEXT_SUPPRESSION_FACTOR` of 0.50 is multiplied against the `sudden_event_score`.
Persistent faults receive NO suppression, as a stuck sensor is a fault regardless of meteorological context.

## 4. Confidence vs Score
Anomaly Confidence is mathematically distinct from Anomaly Score. 
While Score represents the raw magnitude of the evidence, Confidence measures the structural agreement:
`confidence = 0.5 * score + 0.5 * agreement_factor`
This ensures that a single extreme outlier with no supporting corroboration yields moderate confidence, whereas a multivariate-supported fault yields high confidence.

## 5. Fault Hints and Classifications
The winning fault family deterministically defines the classification hint:
- `DATA_LOSS_COMMUNICATION` -> `MISSING`
- `PERSISTENT_SENSOR_FAULT` -> `FROZEN` or `DRIFT` depending on stability vs drift evidence.
- `SUDDEN_EVENT` -> `SPIKE`, `OFFSET`, `NOISE`, or `MULTIVARIATE_INCONSISTENCY` depending on rate vs statistical vs disagreement ratios.

## 6. Thresholding
Development thresholds are derived by taking the 99.5th percentile of the Sudden and Persistent scores on the development dataset. (Note: Because the development dataset contains synthetic anomalies, if anomalies easily reach 1.0, the 99.5th percentile becomes 1.0, heavily suppressing recall to prioritize high precision).
