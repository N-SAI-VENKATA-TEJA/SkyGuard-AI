# SkyGuard AI

## 1. Problem
**SIH Problem Statement:** SIH26073 — AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

SkyGuard AI monitors real-time telemetry from Automatic Weather Stations (AWS) to identify suspicious sensor behavior and physical anomalies. The core objective is to detect anomalies in essential meteorological parameters, specifically: Temperature, Atmospheric Pressure, and Relative Humidity. It is an operational sensor integrity system, not a weather forecasting system.

## 2. Solution
The solution is a complete streaming ML architecture:
AWS observations → causal feature extraction → multiple anomaly detectors → Hybrid V2 evidence fusion → sensor health tracking → API/WebSocket → dashboard.

Crucially, the system is designed to distinguish unusual weather observations from likely sensor hardware or data-quality problems by intersecting temporal, statistical, and multivariate evidence. It mathematically suppresses natural extreme weather transitions while flagging physically impossible contradictions.

## 3. Key Capabilities
- Real-time observation processing
- Spike detection
- Drift detection
- Frozen/stuck sensor detection
- Offset detection
- Multivariate inconsistency detection
- Missing/data-loss detection
- Severity grading
- Confidence approximation
- Deterministic English explanation
- Stateful sensor health tracking
- Maintenance recommendation emission
- Multi-station in-process isolation
- Live dashboard via WebSocket streaming

*(Note: Detection of highly subtle, low-variance noise remains a documented limitation as it organically evades standard statistical thresholds.)*

## 4. Architecture
```text
AWS Observation
       ↓
    FastAPI
       ↓
Station-specific StreamPipeline
       ↓
Causal Feature Engineering
       ↓
Frozen ML Detectors
       ↓
Hybrid V2 Fusion Engine
       ↓
Sensor Health Tracker
       ↓
API Response + WebSocket
       ↓
   Dashboard
```

## 5. Data
The primary development base is the Max Planck Weather Dataset, providing historical weather observations sampled at 10-minute intervals containing temperature, pressure, and relative humidity.

Because massive, perfectly-labeled open-source datasets of specific meteorological hardware failures do not exist, synthetic anomalies (Spikes, Drifts, Freezes, Offsets) were mathematically injected strictly for controlled evaluation. Synthetic labels represent simulated hardware failure boundaries and do not constitute naturally observed ground truth.

## 6. Data Preparation
The raw data underwent strict QA:
- Duplicate timestamps were safely removed.
- Chronological ordering was enforced.
- Physical validity bounds (e.g., Humidity 0-100%) were checked (no violations).
- Natural meteorological outliers were strictly preserved without arbitrary deletion.
- No interpolation was used during the primary cleaning step to avoid manufacturing false data.

## 7. Anomaly Simulation
To establish verifiable ground-truth bounding boxes for algorithm evaluation, we built a synthetic injection framework. The system injects simulated faults representing:
- Spike (Transient electronic noise)
- Drift (Gradual calibration decay)
- Frozen (Stuck sensor reading)
- Offset (Persistent physical bias)
- Noise (Interference)
- Missing (Communication loss)
- Multivariate Inconsistency (Cross-variable fault)
These act as controlled test injections.

## 8. Feature Engineering
The feature engineering pipeline extracts 89 leakage-safe, strictly causal features including:
- Raw sensor values
- Temporal changes/rates
- Causal rolling statistics (e.g., 60-minute rolling variance)
- Stability/frozen behavior (e.g., consecutive unchanged values)
- Multivariate consistency (e.g., z-score divergence between sensors)
- Cross-sensor temporal relationships
- Missing/data-loss indicators
- Robust statistics (IQR, Medians)

All features are strictly causal, utilizing only current and past information.

## 9. ML Detection
The system employs three baseline detectors:
- **Statistical Baseline:** Tracks rolling physical thresholds.
- **PCA:** Analyzes multivariate reconstruction errors.
- **Isolation Forest:** Maps structural density.

**Hybrid V2:**
Because no single baseline detector can handle all failure modes without massive false positives during natural weather transitions, Hybrid V2 fuses evidence from:
- Temporal behavior
- Statistical deviation
- Multivariate inconsistency
- Model support
- Contextual suppression
- Persistent-fault evidence
- Communication/data-loss evidence

The system utilizes deterministic ML engineering and explicit evidence intersection. It is not a deep learning black box.

## 10. Sensor Health
Anomaly detection and sensor health are mathematically decoupled concepts. 
- Anomaly detection is a point-in-time contextual event discovery.
- Sensor health is a persistent, exponentially decaying memory of anomalies over time.

The system maintains a per-sensor health score bounded between 0 and 100. Anomalies trigger health degradation via persistence thresholds, dropping the status through `HEALTHY` → `WATCH` → `DEGRADED` → `CRITICAL`. Gradual recovery occurs only when normal observations resume.

## 11. Real-Time Architecture
The system utilizes a per-station isolated state. Causal history is buffered in memory, driving streaming feature generation which feeds inference. The results are returned via API and broadcast via WebSocket to the dashboard.
- Measured StreamPipeline latency: ~47.11 ms / observation
- Measured API latency: ~83.36 ms / observation
*(These are measured local benchmark values, not universal production guarantees).*

## 12. Dashboard
The React-based web dashboard connects via WebSocket to display:
- Live T/P/RH charts
- Overall anomaly state
- Anomaly score & Severity
- Confidence
- Fault type & Affected sensor
- Deterministic explanation
- Data quality & Sensor health gauges
- Event timeline
- Multi-station selection

## 13. Evaluation
- **Event-level recall:** 90.77%
- **Row-level precision:** 60.14%

These metrics were obtained on the project's synthetic evaluation dataset using the defined evaluation methodology. The architecture was also successfully validated against realistic historical normal-data to confirm it accurately suppresses false alarms during natural extreme weather.

## 14. Demo Scenarios
The final validation (`Step 11`) provides fully reproducible evidence for:
- Normal dynamic weather processing
- Temperature spike
- Sensor drift
- Frozen sensor
- Sensor offset
- Multivariate inconsistency
- Complete data loss
- Sensor-health degradation
- Multi-station isolation

Results are logged in `docs/step11_demo_evidence.md`.

## 15. Limitations
- Mild noise may evade detection natively.
- Evaluated against a synthetic anomaly ground truth.
- No neighboring-station spatial validation implemented.
- No user-facing corrected/imputed values broadcasted.
- Relies on in-memory station state.
- Distributed deployment is not implemented (requires external state store).
- Energy benchmark not measured.
- Browser E2E latency not measured.
- Natural extreme weather may still produce transient anomaly indications.
- The system does not prove with absolute certainty whether an extreme meteorological event is genuine.

## 16. Future Scope
*FUTURE WORK:*
- Neighboring-station spatial consistency validation.
- Statistically calibrated confidence scoring.
- Evaluation against real AWS historical fault datasets.
- User-facing corrected/imputed observation feeds.
- Persistent external state (Redis) for distributed deployment.
- Hardware/power evaluation.
- Stronger mild-noise characterization.

## 17. Running the Project

### Prerequisites
Install the required packages. Key dependencies include:
`fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `websocket-client`, `httpx`

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib websocket-client httpx
```

### Backend
Start the FastAPI + WebSocket server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

For the API performance demo (streams 50 observations and benchmarks latency):
```bash
python api/run_api_demo.py
```

### Dashboard
Install dependencies and start the Vite development server:
```bash
cd dashboard
npm install
npm run dev
```

### Tests
Execute the regression testing suites:
```bash
pytest tests/test_step9_end_to_end.py tests/test_step91.py tests/test_step92.py
```

### Demo
Generate the final demo scenario evidence (streams all 9 scenarios through the live API):
```bash
python tests/test_step11_demo_scenarios.py
```

## 18. Project Structure
```text
SkyGuard-AI/
├── api/                   # FastAPI backend, WebSocket manager, Schemas
├── dashboard/             # React web application frontend
├── data/
│   ├── artifacts/         # Frozen ML models and baselines
│   ├── processed/         # Cleaned and generated CSV datasets
│   └── raw/               # Original weather dataset
├── docs/                  # Technical documentation and evaluation reports
├── ml/
│   ├── anomaly_engine/    # Hybrid V2 logic, Sensor Health tracker
│   ├── features/          # Causal feature engineering and pipelines
│   └── models/            # Baseline ML model training scripts
├── simulator/             # Synthetic anomaly injection and dataset generation
├── tests/                 # Integration, unit, and validation suites
└── README.md
```
