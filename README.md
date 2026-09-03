# SkyGuard AI

**SIH Problem Statement:** SIH26073 — AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

## Overview
SkyGuard AI is an intelligent real-time anomaly detection system designed for Automatic Weather Stations (AWS). Its core objective is to detect anomalies in essential meteorological parameters, specifically:
- Temperature
- Atmospheric Pressure
- Relative Humidity

## Current Implementation Status
**Stage:** STEP 4 — Feature Engineering & Temporal Context

In this stage, we have engineered causal, leakage-safe temporal and multivariate features for our sensor data.

- [x] **STEP 1**: Data auditing and QA
- [x] **STEP 2**: Data cleaning and normalization
- [x] **STEP 3**: Synthetic anomaly injection framework
- [x] **STEP 4**: Feature Engineering & Temporal Context
- [ ] **STEP 5**: Unsupervised anomaly detection baseline

*   **Raw Dataset:** `data/raw/max_planck_weather_ts.csv` (Untouched, read-only)
*   **Processed Baseline Dataset:** `data/processed/aws_clean.csv` (Cleaned, sorted base data)
*   **Generated Datasets:** 
    *   `aws_dev_synthetic.csv` (Small dataset for rapid development, ~8% anomalies)
    *   `aws_synthetic_anomalies.csv` (Full evaluation dataset, ~5% anomalies)
    *   `labels_*.csv` (Event-level ground truth definitions)
*   **Documentation:** Detailed descriptions of anomaly models and configurations are available in [`docs/anomaly_injection.md`](file:///c:/Users/saive/SkyGuard-AI/docs/anomaly_injection.md).

**Anomaly Types Available:**
- Spike (Sudden noise)
- Drift (Gradual calibration decay)
- Frozen (Stuck sensor)
- Offset (Persistent physical bias)
- Abnormal Noise (Interference)
- Missing Data (Communication loss)
- Multivariate Inconsistency (Cross-variable fault)

*(Note: We have **NOT** implemented the ML anomaly detector yet. The current framework purely generates the target data for the detector to eventually learn from).*

## Running the Injection Pipeline
To generate the datasets locally:
```bash
python simulator/generate_dataset.py --dev
python simulator/generate_dataset.py
```

To validate dataset integrity and generate sample visual plots:
```bash
python simulator/validate_dataset.py
```
