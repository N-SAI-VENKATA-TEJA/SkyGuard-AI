# Data Quality Report (Step 2)

## 1. Dataset Overview
This report summarizes the data quality analysis and safe preprocessing steps applied to the Max Planck Weather dataset. The goal of this step was to produce a safe, chronologically ordered, analysis-ready dataset while strictly preserving the original observations, anomalies, and potential sensor faults for downstream modeling.

## 2. Raw Dataset Size
*   **Initial size:** 420,551 rows
*   **Features preserved:** 4 (`timestamp`, `temperature`, `pressure`, `humidity`)

## 3. Time Period
*   **Start Time:** 2009-01-01 00:10:00
*   **End Time:** 2017-01-01 00:00:00

## 4. Sampling Interval
*   **Expected Interval:** 10 minutes
*   **Actual Intervals:** Predominantly 10 minutes, but irregular gaps exist.

## 5. Missing Values
*   There are **0 missing values (`NaN`)** in the core columns.

## 6 & 7. Duplicate Rows and Timestamps
*   **Exact Duplicate Rows:** 327
*   **Duplicate Timestamps (Total):** 327
*   **Conflicting Duplicates:** 0. (All duplicate timestamps corresponded to exactly identical measurements across all sensors).
*   **Action taken:** The 327 exact duplicate rows were safely dropped to prevent dataset bloat and model bias, while ensuring no data was lost. 

## 8. Timestamp Ordering
*   **Original State:** The dataset contained timestamps that were not strictly chronologically ordered (due to the presence and placement of the duplicate rows).
*   **Processed State:** After removing exact duplicates, the dataset was confirmed to be perfectly monotonic and strictly chronologically sorted.

## 9. Major Time Gaps
Significant communication or operational outages were identified in the data. The largest gaps include:
1.  **3 days 02:20:00** (2016-10-25 10:30:00 to 2016-10-28 12:50:00) — approx. 445 missing 10-min observations.
2.  **16 hours** (2014-09-24 17:00:00 to 2014-09-25 09:00:00) — approx. 95 missing 10-min observations.
3.  **30 minutes** (2009-10-08 09:40:00 to 2009-10-08 10:10:00) — approx. 2 missing 10-min observations.

*   **Action taken:** These gaps were strictly preserved. No interpolation was performed, as the anomaly detection engine will need to handle or learn from missing-data faults.

## 10. Physical Range Checks
*   **Temperature:** Min = -23.01°C, Max = 37.28°C
*   **Pressure:** Min = 913.60 mbar, Max = 1015.35 mbar
*   **Relative Humidity:** Min = 12.95%, Max = 100.00%
*   **Result:** All values fall within broad physical constraints. No impossible values (e.g., negative Kelvin equivalent, RH < 0) were found.

## 11 & 12. Constant-Value Runs (Suspicious Persistence)
Analysis of consecutive identical readings revealed potentially suspicious persistence in sensor outputs.
*   **Temperature:** Longest constant run is 14 readings (approx. 2 hours).
*   **Pressure:** Longest constant run is 13 readings.
*   **Relative Humidity (RH):** A massive constant-value run of **239 consecutive readings** was identified (approx. 40 hours stuck at the exact same percentage).
*   **Important Finding:** A long constant-value RH run was identified and retained for subsequent anomaly analysis. Its cause (e.g. sensor saturation, physical fault) is not established during preprocessing.

## 13. Processing Operations Performed
*   Extracted required columns (`timestamp`, `temperature`, `pressure`, `humidity`).
*   Parsed timestamps into robust datetime objects.
*   Dropped 327 identical duplicate rows.
*   Verified and ensured strict chronological sorting.
*   Standardized column names for programmatic access.
*   Saved to a clean CSV: `data/processed/aws_clean.csv`.

## 14. What Was Intentionally NOT Changed
*   **No outliers or extreme values were removed.**
*   **No gaps were interpolated.**
*   **No variables were normalized, scaled, or transformed.**
*   **No anomalies were labeled or removed.** (e.g., the massive RH run is fully preserved).
*   **The original raw CSV (`data/raw/max_planck_weather_ts.csv`) was completely untouched.**

## 15. Known Limitations
*   The gap analysis indicates missing operational periods that any time-series model (e.g., rolling windows) will need to handle explicitly.
*   The persistence runs (like the 40-hour RH run) remain in the dataset. Any statistical metric sensitive to variance drops will naturally flag this region.
