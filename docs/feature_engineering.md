# Step 4: Feature Engineering & Temporal Context

## Overview
This document outlines the robust, leakage-safe feature engineering pipeline created for the SkyGuard AI project. The primary goal is to convert raw weather station observations into temporal, statistical, and multivariate context features without exposing future data to the current observation (causal features).

## Feature Groups

### Group 1: Time / Sampling Features
Features that encode temporal context and gaps.
- `hour`, `minute`, `day_of_week`, `day_of_year`, `month`, `is_weekend`
- Cyclical representations: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
- Gap monitoring: `time_gap_seconds`, `sampling_interval_mins`, `time_gap_flag` (indicates > 15m gap)

### Group 2: Raw Sensor Features
The untransformed data: `temperature`, `pressure`, `humidity`.

### Group 3: First-Order Temporal Features
Capturing immediate temporal shifts.
- `{sensor}_delta_1`: Difference from the immediate previous observation.
- `{sensor}_abs_delta_1`: Absolute difference.
- `{sensor}_rate_per_hour`: Gap-adjusted rate of change per hour.

### Group 4: Causal Rolling Statistics
Strictly causal rolling statistics using past data (`closed='left'`).
Windows: 30m, 60m, 180m, 360m.
- `{sensor}_roll_mean_{w}`
- `{sensor}_roll_std_{w}`
- `{sensor}_roll_min_{w}`, `{sensor}_roll_max_{w}`, `{sensor}_roll_median_{w}`
- Deviations from rolling baselines: `{sensor}_dev_mean_{w}`, `{sensor}_dev_med_{w}`
- Rolling Z-Score: `{sensor}_roll_z_{w}` (computed with safe handling of 0 std deviation by substituting 1e-6).

### Group 5: Stability / Frozen Sensor Features
Features designed to detect static (frozen) sensors.
- `{sensor}_consec_unchanged`: Streak counter of consecutive near-zero absolute changes.
- `{sensor}_roll_var_60m`: Rolling variance over 60m.
- `{sensor}_roll_range_60m`: Rolling peak-to-peak range over 60m.

### Group 6: Multivariate Consistency Features
Cross-sensor relationships.
- `multivariate_z_disagreement`: The maximum absolute difference between Z-scores of temperature, pressure, and humidity.
- `num_sensors_large_dev`: Count of sensors with $|Z| > 2$.
- `dominant_sensor_z`: Indicator for the sensor exhibiting the most extreme standardized deviation.

### Group 7: Cross-Sensor Temporal Comparisons
- `temp_rate_ratio`, `press_rate_ratio`, `humid_rate_ratio`: The relative magnitude of a sensor's rate of change compared to the total absolute change across all sensors.

### Group 8: Missing / Communication Features
- `{sensor}_missing`: 1 if sensor is NaN, else 0.
- `any_sensor_missing`: 1 if any sensor is missing.
- `all_sensors_missing`: 1 if all sensors are missing.

### Group 9: Sensor-Specific Robust Features
- `{sensor}_roll_iqr_60m`: The Interquartile Range (IQR) over a 60m causal rolling window.
- `{sensor}_robust_z_60m`: Deviation from rolling median normalized by rolling IQR.

## Causal Leakage Prevention
All rolling statistics are strictly past-only using `rolling(window, closed='left')`. 
A rigorous testing framework (`run_leakage_test`) verifies that at multiple timestamp injection points, appending future records does not alter the feature vector generated for time $t$.

## Missing Value Handling
- **Raw missing values**: Intentionally retained and flagged (Group 8) to model communication failures.
- **Rolling Window Warm-up**: The first 360 minutes of the dataset exhibit expected NaNs in rolling features. 
- **Zero Standard Deviation**: Z-scores handle identical records safely by injecting a micro-variance term (1e-6) to prevent unexpected infinities.

## Reproduction
To run the full feature engineering pipeline:
```bash
python ml/features/run_features.py
```
Outputs are written to `data/processed/aws_dev_features.csv` and `data/processed/aws_synthetic_features.csv`.
Validation plots are saved to `docs/validation/features/`.
