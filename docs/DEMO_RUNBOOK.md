# SkyGuard AI: Demo Runbook

This runbook provides step-by-step instructions for demonstrating the system to evaluators. It is designed to be executed using the automated Python scripts or via the React Dashboard.

**Preparation:**
1. Start the backend: `uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`
2. Start the frontend dashboard: `cd dashboard && npm run dev`
3. Have `tests/test_step11_demo_scenarios.py` ready.

---

### Scenario 1: Normal Dynamic Weather
- **Action:** Stream sequential observations from `data/processed/aws_clean.csv`.
- **Audience Expectation:** The dashboard charts plot normally without generating continuous alarms.
- **Presenter Script:** *"The system is currently ingesting real historic chronological weather data. Notice that despite natural shifts in temperature and pressure, the Hybrid engine suppresses false positives because the structural relationship between the variables remains physically coherent."*
- **Evidence Reference:** `docs/step11_demo_evidence.md`

### Scenario 2: Temperature Spike
- **Action:** Inject a sudden +25C magnitude jump into the temperature feed.
- **Audience Expectation:** A `CRITICAL` alert appears within a single API response cycle.
- **System Behavior:** Flagged as `MULTIVARIATE_INCONSISTENCY` (or `SPIKE`).
- **Presenter Script:** *"A sudden, physically impossible temperature spike just hit the sensor. The system detects that pressure and humidity did not respond accordingly—indicating a sensor hardware error, not a freak weather event."*

### Scenario 3: Sensor Offset
- **Action:** Introduce a persistent offset to the temperature sensor (+5C shift) while allowing natural variance to continue.
- **Audience Expectation:** The anomaly is detected shortly after the injection.
- **System Behavior:** Flagged as an `OFFSET` or `MULTIVARIATE_INCONSISTENCY`.
- **Presenter Script:** *"Here we introduce a subtle calibration offset. A static threshold would miss this if it stayed within seasonal norms, but our dynamic historical baseline captures the structural deviation immediately."*

### Scenario 4: Complete Data Loss
- **Action:** Send a JSON payload containing `null` for Temperature, Pressure, and Humidity.
- **Audience Expectation:** Anomaly triggers explicitly as `DATA_LOSS`.
- **System Behavior:** API successfully processes the nulls, routing them to the data-quality tracker.
- **Presenter Script:** *"Communication just went down. Note how the system correctly flags this as a data-quality failure, rather than penalizing the physical hardware health score of the sensors."*

### Scenario 5: Frozen Sensor & Health Decay
- **Action:** Send exactly identical readings for all sensors for 40 consecutive intervals (6+ hours).
- **Audience Expectation:** The anomaly triggers as `FROZEN`, and over time, the Sensor Health gauge falls from `HEALTHY` to `CRITICAL`.
- **Presenter Script:** *"The sensors have frozen. The anomaly detector flags the zero-variance flatline. Watch the Sensor Health gauges—because this fault is persisting over multiple hours, the exponential penalty begins degrading the operational health of the station, eventually issuing a maintenance recommendation."*

### Scenario 6: Multi-Station Isolation
- **Action:** Switch the dashboard view between `AWS_STATION_A` and `AWS_STATION_B` while sending interleaved payloads.
- **Audience Expectation:** The streams do not cross-talk; health and anomaly states are completely separate.
- **Presenter Script:** *"We are processing multiple stations concurrently on the same backend. The dynamic memory dictionaries ensure zero state leakage between physical locations."*
