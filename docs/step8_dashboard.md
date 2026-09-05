# Step 8.4: Live Dashboard Architecture

## 1. Overview
The Live Anomaly Detection Dashboard provides a real-time, purely presentation-layer frontend designed for live monitoring and demonstration of the SkyGuard AI ML backend. It visualizes the stateful output of the Step 8.3 FastAPI + WebSocket implementation without duplicating any intelligence logic.

## 2. Frontend Technology Stack
- **Framework**: React 18
- **Build Tool**: Vite
- **Charting**: Recharts (for live bounded time-series)
- **Icons**: Lucide React
- **Styling**: Vanilla CSS using Grid/Flexbox.

## 3. Data Contract Integration
The dashboard strictly consumes the `ObservationResponse` dictionary via the WebSocket endpoint `/ws/{station_id}`.

**Fields Consumed Directly from Backend:**
- `timestamp`: Formatted to local time for charts and timelines.
- `temperature`, `pressure`, `humidity`: Plotted on the Recharts graph and displayed on sensor cards.
- `processing_state`: Displayed in anomaly summary.
- `anomaly_flag`: Drives the primary UI state (Normal / Anomalous banner colors) and event timeline.
- `anomaly_score`, `severity`, `confidence`: Displayed natively without frontend re-calculation.
- `fault_type`, `affected_sensor`, `explanation`: Rendered directly in timeline anomaly events.
- `sensor_health_temperature`, `sensor_health_pressure`, `sensor_health_humidity`: Passed explicitly from Step 7 logic.
- `temperature_status`, `pressure_status`, `humidity_status`: Drives the color coding (Healthy/Degraded/Critical) of the health dials.
- `data_quality_status`: Displayed natively.

## 4. Key UI Components
### Header & Controls
- Displays connection indicator natively tied to WebSocket states (`onopen`, `onclose`, `onerror`).
- Contains a Station Selector (`AWS_DEMO_01`, `AWS_002`) allowing seamless switching between isolated backend pipeline streams.

### Overall Anomaly Status
- Banners switch dynamically (Green / Red) based strictly on `anomaly_flag`. No frontend threshold logic is applied.

### Sensor Observation Cards & Health Grid
- Displays precise numeric values and step 7 continuity health scores side-by-side.

### Bounded Live Elements
To preserve browser memory in continuous real-time execution:
- **Live Chart**: Bounded to the most recent `MAX_HISTORY = 30` observations.
- **Event Timeline**: Only unshifts events when `anomaly_flag` is explicitly true, bounded to `MAX_EVENTS = 20`.

## 5. Station Handling & Reconnection
- **Station Switching**: When a new station is selected, the existing WebSocket connection is explicitly terminated, the local React state (charts and history) is completely purged, and a new socket is opened to `/ws/{new_station_id}`.
- **Reconnection**: Handled via simple recursive timeout on the `onclose` event handler (3000ms backoff).

## 6. Demo Procedure
1. Start the backend: `python -m uvicorn api.main:app`
2. Start the frontend: `cd dashboard && npm run dev`
3. Execute the simulator: `python api/run_api_demo.py` (which pushes the Max Planck dataset rapidly through the POST endpoint).
4. The dashboard will natively intercept the broadcasted WS packets for the corresponding `station_id` and visualize the telemetry in real-time.

## 7. Limitations
- Dashboard stores only bounded in-memory display history.
- Browser refresh clears the visual history.
- Backend state is currently in-process.
- Distributed deployment has not been implemented.
- WebSocket end-to-end latency has not been separately benchmarked.
- Data currently comes from simulated AWS observations.
- Spatial neighboring-station consistency is not implemented.
- Corrected/imputed-value visualization is not currently implemented.
