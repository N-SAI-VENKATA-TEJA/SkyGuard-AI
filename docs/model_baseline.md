# Step 5: Unsupervised Anomaly Detection Baseline

## Overview
This document details the configuration and evaluation of three unsupervised baseline anomaly detection models designed to establish benchmark performance on the synthetic SkyGuard AI dataset.

## Training & Preprocessing Methodology
- **Data Segregation**: Models were fitted and calibrated strictly on the development dataset (`aws_dev_features.csv`). Evaluation occurred chronologically on the full evaluation dataset (`aws_synthetic_features.csv`).
- **Feature Selection**: Administrative/label columns and `timestamp` were excluded. All causal temporal, stability, multivariate, missing indicators, and gap features from Step 4 were retained.
- **Missing Value Handling**: A `SimpleImputer(strategy='median')` was fitted exclusively on the development set to safely impute NaN values required by PCA and Isolation Forest. The explicit `_missing` indicator features were retained.
- **Scaling**: A `StandardScaler` was applied to numerical features, fitted exclusively on the development set.
- **Contamination Disclosure**: The development dataset contains approximately 5% synthetic anomalies. The models were fitted in an unsupervised fashion on this entire set without label filtering.

## Models
### 1. Statistical Baseline
- **Configuration**: Uses the maximum causal 60-minute standardized deviation across temperature, pressure, and humidity.
- **Formula**: `max(abs(temperature_roll_z_60m), abs(pressure_roll_z_60m), abs(humidity_roll_z_60m))`
- **Thresholding**: Static threshold set to the 99.5th percentile of the statistical score distribution in the development dataset.

### 2. PCA Multivariate Detector
- **Configuration**: Principal Component Analysis fitted to explain 95% of the cumulative variance.
- **Scoring**: Reconstruction Error (Mean Squared Error).
- **Thresholding**: Static threshold set to the 99.5th percentile of reconstruction errors in the development dataset.

### 3. Isolation Forest
- **Configuration**: `n_estimators=200`, `random_state=42`, `contamination='auto'`.
- **Scoring**: Native `score_samples()` was extracted and inverted (multiplied by -1) to conform to the standard where a higher score indicates greater anomalousness.
- **Thresholding**: Static threshold set to the 99.5th percentile of the inverted scores in the development dataset. Native `.predict()` was ignored.

## Evaluation Protocol
- **Event-Level Metric**: An anomaly event is considered detected if *at least one row* belonging to that event's `anomaly_id` is correctly flagged.
- **Temporal Integrity**: The full evaluation dataset was processed in its original chronological order without shuffling.

## Important Limitations
The anomalies in the evaluation dataset are mathematically injected synthetics. While they simulate hardware failure behaviors, high performance on this baseline does not guarantee identical performance on organic AWS hardware faults. These models serve as comparative benchmarks for future supervised/advanced architectures.
