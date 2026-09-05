import pytest
from fastapi.testclient import TestClient
from api.main import app, pipelines, last_timestamps, station_locks
from ml.anomaly_engine.stream_pipeline import StreamPipeline
import pandas as pd
import time

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_state():
    """Reset the global state before each test."""
    pipelines.clear()
    last_timestamps.clear()
    station_locks.clear()
    yield

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "SkyGuard AI"}

def test_observation_invalid_payload():
    response = client.post("/api/v1/observations", json={
        "station_id": "AWS_001",
        "temperature": 20.0
        # Missing required fields
    })
    assert response.status_code == 422 # Unprocessable Entity

def test_observation_invalid_timestamp():
    response = client.post("/api/v1/observations", json={
        "station_id": "AWS_001",
        "timestamp": "invalid-timestamp",
        "temperature": 20.0,
        "pressure": 1010.0,
        "humidity": 50.0
    })
    assert response.status_code == 400
    assert "Invalid timestamp format" in response.json()["detail"]

def test_observation_normal():
    payload = {
        "station_id": "AWS_001",
        "timestamp": "2026-09-04T12:00:00",
        "temperature": 20.0,
        "pressure": 1010.0,
        "humidity": 50.0
    }
    response = client.post("/api/v1/observations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "AWS_001"
    assert data["processing_state"] == "WARMUP"
    assert data["anomaly_flag"] is False

def test_station_isolation():
    # Send observation to AWS_001
    client.post("/api/v1/observations", json={
        "station_id": "AWS_001",
        "timestamp": "2026-09-04T12:00:00",
        "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
    })
    
    # Send observation to AWS_002
    client.post("/api/v1/observations", json={
        "station_id": "AWS_002",
        "timestamp": "2026-09-04T12:00:00",
        "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
    })
    
    assert "AWS_001" in pipelines
    assert "AWS_002" in pipelines
    assert pipelines["AWS_001"] is not pipelines["AWS_002"]

def test_ordering():
    # Send first observation
    client.post("/api/v1/observations", json={
        "station_id": "AWS_001",
        "timestamp": "2026-09-04T12:10:00",
        "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
    })
    
    # Send older observation
    response = client.post("/api/v1/observations", json={
        "station_id": "AWS_001",
        "timestamp": "2026-09-04T12:00:00", # Older
        "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
    })
    assert response.status_code == 400
    assert "Out-of-order observation" in response.json()["detail"]

def test_state_continuity():
    station_id = "AWS_STATE"
    ts = pd.Timestamp("2026-01-01T00:00:00")
    
    # Send 40 rows to exit warmup
    for i in range(40):
        ts += pd.Timedelta(minutes=10)
        res = client.post("/api/v1/observations", json={
            "station_id": station_id,
            "timestamp": ts.isoformat(),
            "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
        })
        assert res.status_code == 200
        
    data = res.json()
    assert data["processing_state"] == "PROCESSED"
    assert data["temperature_status"] == "HEALTHY"
    
    # Send anomaly
    ts += pd.Timedelta(minutes=10)
    res_anom = client.post("/api/v1/observations", json={
        "station_id": station_id,
        "timestamp": ts.isoformat(),
        "temperature": 40.0, "pressure": 1010.0, "humidity": 50.0 # Anomaly spike
    })
    
    data_anom = res_anom.json()
    assert data_anom["anomaly_flag"] is True
    assert data_anom["sensor_health_temperature"] < 100.0

def test_api_vs_stream_equivalence():
    # Run StreamPipeline directly
    direct_pipeline = StreamPipeline()
    df = pd.read_csv('data/processed/aws_clean.csv', nrows=50)
    
    direct_results = []
    for i, row in df.iterrows():
        res = direct_pipeline.process_observation(
            timestamp=pd.Timestamp(row['timestamp']),
            temperature=row['temperature'],
            pressure=row['pressure'],
            humidity=row['humidity']
        )
        direct_results.append(res)
        
    # Run through API
    api_results = []
    for i, row in df.iterrows():
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_EQUI",
            "timestamp": row['timestamp'],
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        })
        api_results.append(res.json())
        
    # Compare
    for d_res, a_res in zip(direct_results, api_results):
        assert d_res['anomaly_score'] == a_res['anomaly_score']
        assert d_res['anomaly_flag'] == a_res['anomaly_flag']
        assert d_res['sensor_health_temperature'] == a_res['sensor_health_temperature']
        assert d_res['processing_state'] == a_res['processing_state']

def test_websocket():
    with client.websocket_connect("/ws/AWS_WS") as websocket:
        # Trigger event
        response = client.post("/api/v1/observations", json={
            "station_id": "AWS_WS",
            "timestamp": "2026-09-04T12:00:00",
            "temperature": 20.0,
            "pressure": 1010.0,
            "humidity": 50.0
        })
        assert response.status_code == 200
        
        # Receive websocket data
        data = websocket.receive_json()
        assert data["station_id"] == "AWS_WS"
        assert data["temperature"] == 20.0
        
def test_websocket_isolation():
    with client.websocket_connect("/ws/AWS_001") as ws_1:
        with client.websocket_connect("/ws/AWS_002") as ws_2:
            # Send to AWS_001
            client.post("/api/v1/observations", json={
                "station_id": "AWS_001",
                "timestamp": "2026-09-04T12:00:00",
                "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
            })
            
            # ws_1 should receive it
            data1 = ws_1.receive_json()
            assert data1["station_id"] == "AWS_001"
            
            # Send to AWS_002
            client.post("/api/v1/observations", json={
                "station_id": "AWS_002",
                "timestamp": "2026-09-04T12:00:00",
                "temperature": 20.0, "pressure": 1010.0, "humidity": 50.0
            })
            
            # ws_2 should receive it (ws_1 shouldn't receive AWS_002 data)
            data2 = ws_2.receive_json()
            assert data2["station_id"] == "AWS_002"
