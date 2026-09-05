import { useState, useEffect, useRef } from 'react';
import { Activity, Thermometer, Gauge, Droplets, AlertTriangle, CheckCircle, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Limit historical points for charts and events
const MAX_HISTORY = 30;
const MAX_EVENTS = 20;

function App() {
  const [stationId, setStationId] = useState('AWS_DEMO_01');
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // connected, disconnected, connecting
  const [currentData, setCurrentData] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [events, setEvents] = useState([]);
  
  const wsRef = useRef(null);

  // WebSocket Connection Logic
  useEffect(() => {
    // Clear old data when switching stations
    setCurrentData(null);
    setHistoryData([]);
    setEvents([]);
    setConnectionStatus('connecting');

    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${stationId}`);
      wsRef.current = ws;

      ws.onopen = () => setConnectionStatus('connected');
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Ensure data is for current station (guard against race conditions)
        if (data.station_id !== stationId) return;

        const timeStr = new Date(data.timestamp).toLocaleTimeString();
        const chartPoint = {
          time: timeStr,
          temp: data.temperature,
          press: data.pressure,
          humid: data.humidity,
          isAnomaly: data.anomaly_flag
        };

        setCurrentData(data);
        
        setHistoryData(prev => {
          const next = [...prev, chartPoint];
          if (next.length > MAX_HISTORY) return next.slice(next.length - MAX_HISTORY);
          return next;
        });

        // Add event if it's an anomaly
        if (data.anomaly_flag) {
          setEvents(prev => {
            const newEvent = {
              id: data.timestamp + Math.random(),
              time: timeStr,
              fault: data.fault_type,
              sensor: data.affected_sensor,
              severity: data.severity,
              confidence: data.confidence,
              explanation: data.explanation
            };
            const next = [newEvent, ...prev];
            if (next.length > MAX_EVENTS) return next.slice(0, MAX_EVENTS);
            return next;
          });
        }
      };

      ws.onclose = () => {
        setConnectionStatus('disconnected');
        // Simple reconnect logic
        setTimeout(() => {
          if (wsRef.current === ws) {
            connectWebSocket();
          }
        }, 3000);
      };

      ws.onerror = () => setConnectionStatus('disconnected');
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [stationId]);

  // Render Helpers
  const renderHealthScore = (score, status) => {
    let color = 'var(--status-normal)';
    if (status === 'WARNING' || status === 'WATCH') color = 'var(--status-warning)';
    if (status === 'CRITICAL' || status === 'DEGRADED') color = 'var(--status-danger)';
    
    return (
      <div className="health-score-container">
        <div className="health-score-circle" style={{ borderColor: color, color }}>
          {score ? score.toFixed(0) : '—'}
        </div>
        <div style={{ color, fontWeight: 600 }}>{status || 'UNKNOWN'}</div>
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      
      {/* 1. HEADER */}
      <div className="top-bar">
        <div className="title-group">
          <h1>SkyGuard AI</h1>
          <p>Intelligent AWS Anomaly Detection</p>
        </div>
        
        <div className="controls">
          <select 
            className="station-select"
            value={stationId}
            onChange={(e) => setStationId(e.target.value)}
          >
            <option value="AWS_DEMO_01">AWS_DEMO_01</option>
            <option value="AWS_002">AWS_002 (Simulated)</option>
          </select>

          <div className={`connection-badge ${connectionStatus}`}>
            <div className="status-dot"></div>
            {connectionStatus.toUpperCase()}
          </div>
        </div>
      </div>

      {/* 2. OVERALL ANOMALY STATUS */}
      <div className={`panel anomaly-panel ${currentData?.anomaly_flag ? 'is-anomalous' : 'not-anomalous'}`}>
        <div className="panel-title">System Status</div>
        
        {currentData ? (
          <>
            <div className="anomaly-status-large">
              {currentData.anomaly_flag ? (
                <><AlertTriangle style={{marginRight: 10, verticalAlign: 'text-bottom'}}/> ANOMALY DETECTED</>
              ) : (
                <><CheckCircle style={{marginRight: 10, verticalAlign: 'text-bottom'}}/> SYSTEM NORMAL</>
              )}
            </div>
            
            <div className="anomaly-stats">
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Anomaly Score</span>
                <span className="anomaly-stat-value">{currentData.anomaly_score?.toFixed(4)}</span>
              </div>
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Severity</span>
                <span className="anomaly-stat-value">{currentData.severity}</span>
              </div>
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Confidence</span>
                <span className="anomaly-stat-value">{(currentData.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Fault Type</span>
                <span className="anomaly-stat-value">{currentData.fault_type || 'NONE'}</span>
              </div>
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Data Quality</span>
                <span className="anomaly-stat-value">{currentData.data_quality_status}</span>
              </div>
              <div className="anomaly-stat-item">
                <span className="anomaly-stat-label">Processing State</span>
                <span className="anomaly-stat-value">{currentData.processing_state}</span>
              </div>
            </div>
          </>
        ) : (
          <div style={{ color: 'var(--text-secondary)' }}>Waiting for data...</div>
        )}
      </div>

      {/* 3. SENSOR OBSERVATION CARDS */}
      <div className="metrics-grid">
        <div className="panel">
          <div className="panel-title"><Thermometer size={18} color="var(--color-temp)"/> Temperature</div>
          <p className="metric-value" style={{color: 'var(--color-temp)'}}>
            {currentData ? currentData.temperature.toFixed(2) : '—'} <span className="metric-unit">°C</span>
          </p>
        </div>
        <div className="panel">
          <div className="panel-title"><Gauge size={18} color="var(--color-press)"/> Pressure</div>
          <p className="metric-value" style={{color: 'var(--color-press)'}}>
            {currentData ? currentData.pressure.toFixed(2) : '—'} <span className="metric-unit">hPa</span>
          </p>
        </div>
        <div className="panel">
          <div className="panel-title"><Droplets size={18} color="var(--color-humid)"/> Humidity</div>
          <p className="metric-value" style={{color: 'var(--color-humid)'}}>
            {currentData ? currentData.humidity.toFixed(2) : '—'} <span className="metric-unit">%</span>
          </p>
        </div>
      </div>

      {/* 4. SENSOR HEALTH */}
      <div className="health-grid">
        <div className="panel">
          <div className="panel-title">Temperature Health</div>
          {currentData ? renderHealthScore(currentData.sensor_health_temperature, currentData.temperature_status) : '—'}
        </div>
        <div className="panel">
          <div className="panel-title">Pressure Health</div>
          {currentData ? renderHealthScore(currentData.sensor_health_pressure, currentData.pressure_status) : '—'}
        </div>
        <div className="panel">
          <div className="panel-title">Humidity Health</div>
          {currentData ? renderHealthScore(currentData.sensor_health_humidity, currentData.humidity_status) : '—'}
        </div>
      </div>

      {/* 5. LIVE TIME-SERIES & TIMELINE */}
      <div className="charts-grid">
        <div className="panel">
          <div className="panel-title"><Activity size={18} /> Live Telemetry</div>
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: 8}}/>
                <Line yAxisId="left" type="monotone" dataKey="temp" stroke="var(--color-temp)" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line yAxisId="right" type="monotone" dataKey="press" stroke="var(--color-press)" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line yAxisId="left" type="monotone" dataKey="humid" stroke="var(--color-humid)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title"><AlertTriangle size={18} /> Anomaly Timeline</div>
          <div className="timeline">
            {events.length === 0 ? (
              <div style={{color: 'var(--text-secondary)', padding: '1rem'}}>No anomalies detected.</div>
            ) : (
              events.map((ev) => (
                <div key={ev.id} className={`timeline-event ${ev.severity === 'CRITICAL' ? 'critical' : ''}`}>
                  <div className="event-time">{ev.time}</div>
                  <h4 className="event-title">{ev.fault} in {ev.sensor}</h4>
                  <p className="event-desc">{ev.explanation}</p>
                  <p className="event-desc" style={{marginTop: '0.25rem', fontSize: '0.75rem'}}>
                    Severity: <strong>{ev.severity}</strong> | Confidence: {(ev.confidence*100).toFixed(0)}%
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      
    </div>
  );
}

export default App;
