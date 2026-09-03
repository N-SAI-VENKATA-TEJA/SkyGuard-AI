# SkyGuard AI — Step 2 Engineering Walkthrough

## 1. Executive Summary
Step 2 was tasked with converting the raw Max Planck Weather dataset into a safe, chronologically ordered, and analysis-ready baseline dataset. The goal was to deeply understand data quality issues (duplicates, gaps, persistence) and preprocess the data without altering genuine meteorological observations or anomaly patterns.

Step 2 is fully complete. We successfully extracted the required SIH parameters, resolved out-of-order rows by removing identical duplicates, generated comprehensive exploratory plots, and preserved all extreme/suspicious values. There are no remaining unresolved blocking issues for Step 2.

------------------------------------------------------------

## 2. Starting Project State
Before Step 2 commenced, the repository contained a basic folder skeleton and a preliminary audit script from Step 1.

```text
SkyGuard-AI/
├── data/
│   └── raw/
│       └── max_planck_weather_ts.csv
├── ml/
│   ├── config.py
│   └── preprocessing/
│       └── data_audit.py
├── README.md
└── (other empty scaffolding directories)
```

**Relevant Details:**
- **Raw Dataset Location:** `data/raw/max_planck_weather_ts.csv`
- **Configuration:** `ml/config.py` was established to map core SIH variables.

------------------------------------------------------------

## 3. Step 2 Objectives

- [x] Timestamp quality analyzed — **COMPLETED**
- [x] Duplicate rows investigated — **COMPLETED**
- [x] Duplicate timestamps investigated — **COMPLETED**
- [x] Conflicting duplicates identified if present — **COMPLETED**
- [x] Data chronologically sorted — **COMPLETED**
- [x] Time gaps identified — **COMPLETED**
- [x] Missing values checked — **COMPLETED**
- [x] Physical ranges checked — **COMPLETED**
- [x] Constant-value runs analyzed — **COMPLETED**
- [x] Temporal plots created — **COMPLETED**
- [x] Distribution/relationship analysis created — **COMPLETED**
- [x] Processed dataset generated — **COMPLETED**
- [x] Raw dataset untouched — **COMPLETED**
- [x] No outliers removed — **COMPLETED**
- [x] No interpolation performed — **COMPLETED**
- [x] No anomaly injection performed — **COMPLETED**
- [x] Data quality report created — **COMPLETED**
- [x] Reproducible preprocessing script created — **COMPLETED**
- [x] Pipeline executed successfully — **COMPLETED**
- [x] Verification completed — **COMPLETED**

------------------------------------------------------------

## 4. Dataset Details
**Actual Dataset Used:** Max Planck Weather Dataset
- **Filename:** `max_planck_weather_ts.csv`
- **Location:** `data/raw/`
- **Number of Rows:** 420,551
- **Number of Columns:** 15 originally.
- **Relevant Columns:** `Date Time`, `T (degC)`, `p (mbar)`, `rh (%)`
- **Date Range:** 2009-01-01 00:10:00 to 2017-01-01 00:00:00
- **Sampling Interval:** Approximately 10 minutes.
- **File Format:** CSV
- **Approximate File Size:** ~42 MB

**Reason for Selection:**
(Engineering Reasoning) This dataset provides a long-term, high-frequency multivariate time-series representative of actual AWS environments. (Observed Fact) It contains naturally occurring noise, significant operational gaps, and sustained periods of sensor persistence (constant runs) that mirror real-world sensor communication and physical fault characteristics required by the SIH problem statement.

------------------------------------------------------------

## 5. Raw Data Integrity
The original dataset `data/raw/max_planck_weather_ts.csv` was strictly treated as read-only.

- **Was it modified?** No.
- **Was it copied?** No, it was read directly into memory via Pandas.
- **Was it overwritten?** No.
- **Were rows removed from it?** No.
- **Were values changed?** No.
- **Were columns renamed?** No.
- **Did any preprocessing touch it?** No. 

**Verification:** The preprocessing script explicitly defines the input path as `RAW_DATA_PATH` and the output path as `PROCESSED_DATA_DIR / aws_clean.csv`. No `to_csv` or file-write commands target the `data/raw/` directory.

------------------------------------------------------------

## 6. Timestamp Analysis
- **Timestamp parsing:** Parsed using `pd.to_datetime()` utilizing the `format="%d.%m.%Y %H:%M:%S"` specification.
- **Original ordering:** The original timestamps were slightly out of order due to duplicate rows being appended or interspersed incorrectly. (Evidenced by negative time differences during initial raw auditing in Step 1).
- **Chronological ordering:** After identical duplicate rows were removed, checking `is_monotonic_increasing` returned `True`, indicating the core data was strictly sequential.
- **Negative time differences:** Resolved completely by duplicate removal.
- **Duplicate timestamps:** 327 instances identified.
- **Zero intervals:** None remained after dropping exact duplicate rows.
- **Common interval:** 10 minutes.
- **Irregular intervals:** Exists natively in the data.
- **Largest gaps:** 
    1. 3 days 02:20:00 (Oct 2016)
    2. 16 hours (Sep 2014)
- **How gaps were handled:** We intentionally did NOT pad, fill, or interpolate these gaps. They are preserved natively.

**Why timestamps were sorted:** Any sequence model, sliding-window algorithm, or differential feature extraction relies on strict chronological sequencing to function correctly. 
**Does sorting change measurements?** No. Reordering rows by time simply places the observations in the physical sequence they occurred; it does not alter the values themselves.

------------------------------------------------------------

## 7. Duplicate Analysis
- **Exact duplicate rows:** 327 rows contained identical data across all 15 columns.
- **Duplicate timestamps:** 327 total duplicate timestamps were identified.
- **Identical values:** All duplicate timestamps contained identical sensor measurements.
- **Conflicting duplicates:** 0 conflicting duplicate timestamps existed.
- **Decision made:** We safely dropped the 327 identical duplicate rows using `df.drop_duplicates()`.
- **Reasoning:** Since the duplicates were perfectly identical, they offer no additional physical information and artificially distort time-series rolling aggregations by creating `0` time-deltas.
- **Before count:** 420,551
- **Removed count:** 327
- **After count:** 420,224

------------------------------------------------------------

## 8. Missing Values and Invalid Values
**DATA VALIDATION** (Not Anomaly Detection)
- **NaN / Missing:** 0 found.
- **Infinite values:** 0 found.
- **Invalid Temperature:** 0 violations (Checked range: -50°C to 60°C).
- **Invalid Pressure:** 0 violations (Checked range: > 0 mbar).
- **Invalid Humidity:** 0 violations (Checked range: 0% to 100%).

All values passed broad physical constraint validation.

------------------------------------------------------------

## 9. Constant-Value Run Analysis
We identified contiguous runs where sensor values did not change at all.

- **Temperature:** Total of 14,204 runs >1. Longest run was 14 consecutive readings (~2.3 hours).
- **Pressure:** Total of 24,513 runs >1. Longest run was 13 consecutive readings (~2.1 hours).
- **Relative Humidity:** Total of 20,548 runs >1. **Longest run was 239 consecutive readings (~40 hours).**

**The Long RH Run:**
This 40-hour block (start: 2013-11-22 09:00:00, end: 2013-11-24 00:40:00) stuck at exactly 100.0% is highly notable. We consider this a **candidate fault pattern** (suspicious persistence).
**Why it was preserved:** In data preprocessing, our goal is to clean structural errors (like duplicate rows). A 40-hour persistent humidity reading, while highly suspicious and likely indicating sensor saturation or physical fault, is a real event captured by the hardware. We must preserve it so our anomaly detection algorithms can learn to identify it automatically.

------------------------------------------------------------

## 10. Exploratory Analysis
We generated visual reports in `docs/plots/`.
- **Time-Series Analysis:** `historical_series.png` and `first_7_days.png` were generated to verify long-term seasonal trends and short-term diurnal cycles.
- **Distributions:** `distributions.png` plots histograms with KDE overlays, confirming physically realistic gaussian-like curves for Temperature and Pressure, and a left-skewed bounded curve for Humidity.
- **Temporal Behaviour:** Confirmed distinct diurnal cycles for temperature and humidity.
- **Correlations:**
    - Temperature vs Humidity: -0.572 (Strong negative correlation, expected physically)
    - Temperature vs Pressure: -0.045
    - Pressure vs Humidity: -0.019

------------------------------------------------------------

## 11. Data Preprocessing Decisions

| Decision | Action Taken | Reason | Impact |
|---|---|---|---|
| **Timestamp conversion** | Parsed `Date Time` to Pandas DateTime | String timestamps cannot be mathematically ordered or diffed. | Enables sorting and gap analysis. |
| **Sorting** | Chronologically sorted data | Requisite for sequential ML models. | Fixed structural negative-time errors. |
| **Duplicate handling** | Dropped 327 exact duplicate rows | Prevents bias in rolling windows and artificial 0-time steps. | Dataset reduced to 420,224 rows. |
| **Gap handling** | Intentionally **NOT** interpolated | Real-world AWS data drops out; ML must handle missing sequences. | Time discontinuities remain in the data. |
| **Outlier handling** | Intentionally **NOT** removed | Our ultimate goal is to detect these; removing them defeats the purpose. | Suspicious runs and extremes remain. |
| **Missing value handling**| None required (0 missing) | N/A | N/A |
| **Column renaming** | Mapped to `timestamp`, `temperature`, `pressure`, `humidity` | English, space-free names prevent programmatic syntax errors. | Clean DataFrame access. |
| **Normalization** | Intentionally **NOT** scaled | Scaling is algorithm-dependent and should happen inside the ML pipeline. | True physical units preserved. |

------------------------------------------------------------

## 12. Processed Dataset
- **Processed filename:** `aws_clean.csv`
- **Location:** `data/processed/aws_clean.csv`
- **Columns:** `timestamp`, `temperature`, `pressure`, `humidity`
- **Row count:** 420,224
- **Sorting state:** Strictly monotonically increasing.
- **Duplicate state:** Zero exact row duplicates. Zero duplicate timestamps.
- **Missing-value state:** Zero NaNs.
- **Gaps remain:** Yes.
- **Suspicious observations remain:** Yes.

**Why this is suitable as a baseline:** It represents the absolute ground-truth physical state of the AWS hardware as received by the server, completely free of structural file errors (like duplicate CSV appends), but still containing all the mechanical/meteorological faults we intend to classify.

------------------------------------------------------------

## 13. Files Created or Modified

| File | Created/Modified | Purpose |
|---|---|---|
| `ml/preprocessing/prepare_dataset.py` | Created | Primary preprocessing pipeline script to clean and output CSV. |
| `ml/preprocessing/exploratory_analysis.py` | Created | EDA script to generate seaborn plots and run-length statistics. |
| `docs/data_quality_report.md` | Created | Detailed report summarizing the dataset's numerical traits and gaps. |
| `docs/plots/*.png` | Created | Visualizations (historical, 7-day, distributions, relationships). |
| `data/processed/aws_clean.csv` | Created | The final analysis-ready dataset. |
| `README.md` | Modified | Updated project status to Step 2. |

------------------------------------------------------------

## 14. Code Architecture
The code is strictly modular:
- **`ml/config.py`**: Central repository for path and column name constants. Both scripts import this.
- **`ml/preprocessing/prepare_dataset.py`**: Loads raw data, filters columns, parses time, drops exact duplicates, sorts, and writes out `aws_clean.csv`.
- **`ml/preprocessing/exploratory_analysis.py`**: A strictly read-only analysis script that reads `aws_clean.csv`, calculates constant-value runs via pandas `cumsum()`, and uses `matplotlib/seaborn` to output PNGs to `docs/plots/`.

------------------------------------------------------------

## 15. Errors Encountered

No meaningful implementation errors occurred.
*(Seaborn background rendering took 1-2 minutes for 420K points due to lack of down-sampling, but completed successfully without crashes.)*

------------------------------------------------------------

## 16. Engineering Decisions

### Decision 1
**Decision:** Drop identical row duplicates, but do not drop or aggregate if sensor values conflicted.
**Alternatives considered:** Silently taking the `mean()` of conflicting duplicate timestamps, or taking the `last()` observation.
**Chosen approach:** `drop_duplicates()` for exact matches; explicit hard-stop if conflicts existed.
**Why:** Identical duplicates are usually artifact of data engineering (appending the same file twice). Conflicting duplicates are sensor hardware errors. 
**Trade-off:** We lost 327 rows of data, but they were perfectly redundant.
**Future implication:** None, as they provided no new information.

### Decision 2
**Decision:** Preserve massive time gaps without padding with `NaN` or interpolating.
**Alternatives considered:** Resampling the dataset to a strict 10-minute frequency and filling missing rows with `NaN` or linear interpolation.
**Chosen approach:** Leave time discontinuities natively.
**Why:** Interpolating 3 days of missing weather data introduces massive synthetic bias. Resampling to add `NaN` rows balloons the dataset unnecessarily.
**Trade-off:** Our ML models (e.g., LSTMs or rolling windows) will need to be explicitly aware of time-deltas, rather than assuming constant step sizes.
**Future implication:** We must engineer time-delta features during feature engineering.

------------------------------------------------------------

## 17. Assumptions
- **Assumption:** The Max Planck dataset timestamp uses local German time or UTC without daylight saving time discontinuities that overlap timestamps.
  - **Why it was necessary:** We sorted by timestamps; if DST fall-back created duplicate local times, our script would treat them as true duplicates.
  - **Confidence level:** High, as no conflicting duplicate timestamps were found.
  - **What could invalidate it:** If the station switched time zones internally mid-dataset.

------------------------------------------------------------

## 18. Current Repository Tree
```text
SkyGuard-AI/
├── README.md
├── data/
│   ├── processed/
│   │   └── aws_clean.csv
│   └── raw/
│       └── max_planck_weather_ts.csv
├── docs/
│   ├── data_quality_report.md
│   └── plots/
│       ├── distributions.png
│       ├── first_7_days.png
│       ├── historical_series.png
│       └── relationships.png
└── ml/
    ├── config.py
    └── preprocessing/
        ├── data_audit.py
        ├── exploratory_analysis.py
        └── prepare_dataset.py
```

------------------------------------------------------------

## 19. Verification Checklist
- [x] Raw dataset preserved
- [x] Timestamp parsed
- [x] Dataset sorted chronologically
- [x] Duplicate rows investigated
- [x] Duplicate timestamps investigated
- [x] Missing values checked
- [x] Physical validity checked
- [x] Time gaps identified
- [x] Constant runs investigated
- [x] Processed dataset created
- [x] Processed dataset verified
- [x] Reproducible preprocessing script works
- [x] Documentation updated

------------------------------------------------------------

## 20. Known Problems / Technical Debt
- **Unusually long gaps:** The 3-day outage in 2016 requires special handling in future feature engineering (e.g., passing `time_since_last_reading` as a feature).
- **Visualization scalability:** Generating line plots for 420K rows is slow. Future visual scripts should implement down-sampling (e.g., hourly averages) before passing to seaborn.

------------------------------------------------------------

## 21. What We Should NOT Do Yet
- We have NOT engineered moving averages, derivatives, or rolling statistics (Feature Engineering).
- We have NOT injected synthetic faults (e.g., spikes, drifts) required for testing models.
- We have NOT implemented Isolation Forests, Autoencoders, or any anomaly scoring.
- We have NOT built APIs, FastAPIs, or frontend dashboard architecture.

------------------------------------------------------------

## 22. Recommended Next Step
**Next Step:** Feature Engineering & Temporal Metrics
- **What it should accomplish:** Extract mathematical representations of the data that highlight anomalies (e.g., rate of change, rolling standard deviations, hour-of-day encodings, time-since-last-reading).
- **Why it comes next:** Machine learning models (like Random Forests or Isolation Forests) perform significantly better when given explicit temporal context (like rolling variance) rather than raw points.
- **Inputs:** `data/processed/aws_clean.csv`
- **Outputs:** `data/processed/aws_features.csv`

------------------------------------------------------------

## 23. Final Status

**STEP 2 STATUS:**
COMPLETE

- **Strongest accomplishment:** Successfully established a perfectly monotonic time-series baseline while actively preserving real-world sensor persistence faults for downstream model training.
- **Biggest remaining concern:** The large data gaps (up to 3 days) will break naïve sliding window aggregations if time-deltas are not explicitly calculated in the next step.
- **Most important fact to know:** The dataset contains a massive 40-hour block where Humidity is locked at exactly 100.0%, serving as our primary target for confirming our future anomaly detector works.
