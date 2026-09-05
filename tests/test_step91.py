import pytest
import pandas as pd
import json
import time
from fastapi.testclient import TestClient
from api.main import app, pipelines, last_timestamps, station_locks
from ml.anomaly_engine.stream_pipeline import StreamPipeline

client = TestClient(app)

def reset_state():
    pipelines.clear()
    last_timestamps.clear()
    station_locks.clear()

def warmup(station):
    for i in range(40):
        client.post("/api/v1/observations", json={
            "station_id": station,
            "timestamp": (pd.Timestamp("2026-01-01T00:00:00") + pd.Timedelta(minutes=10*i)).isoformat(),
            "temperature": 20.0, "pressure": 1012.0, "humidity": 55.0
        })

def test_missing_data_flow():
    reset_state()
    warmup("AWS_MISSING_FLOW")
    res = client.post("/api/v1/observations", json={
        "station_id": "AWS_MISSING_FLOW",
        "timestamp": "2026-01-01T06:40:00",
        "temperature": None,
        "pressure": None,
        "humidity": None
    })
    
    assert res.status_code == 200
    data = res.json()
    assert data["anomaly_flag"] is True
    assert data["fault_type"] == "MISSING"
    assert data["data_quality_status"] == "DATA_LOSS"
    assert data["affected_sensor"] == "ALL_SENSORS"

def test_partial_missing_flow():
    reset_state()
    warmup("AWS_PARTIAL_FLOW")
    res = client.post("/api/v1/observations", json={
        "station_id": "AWS_PARTIAL_FLOW",
        "timestamp": "2026-01-01T06:40:00",
        "temperature": None,
        "pressure": 1012.0,
        "humidity": 55.0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["anomaly_flag"] is True
    assert data["fault_type"] == "MISSING"
    assert data["data_quality_status"] == "DEGRADED"

def test_realistic_normal():
    reset_state()
    # Read from aws_clean.csv
    df = pd.read_csv("data/processed/aws_clean.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').head(50) # use 50 points
    
    res = None
    for _, row in df.iterrows():
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_REAL_NORM",
            "timestamp": row['timestamp'].isoformat(),
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        })
        assert res.status_code == 200
        
    data = res.json()
    # If the realistic normal dataset is clean, we expect False anomaly
    assert data["anomaly_flag"] is False

def test_other_anomalies():
    # Load synthetic anomalies to feed offset, noise, multivariate
    reset_state()
    df = pd.read_csv("data/processed/aws_synthetic_anomalies.csv")
    
    # 1. Offset
    offset_idx = df[df['anomaly_type'] == 'offset'].index
    if len(offset_idx) > 0:
        row = df.loc[offset_idx[0]]
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_OFFSET",
            "timestamp": "2026-01-01T00:00:00",
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        }).json()
        assert res["processing_state"] == "WARMUP" # Just testing processing
        
    # 2. Noise
    noise_idx = df[df['anomaly_type'] == 'noise'].index
    if len(noise_idx) > 0:
        row = df.loc[noise_idx[0]]
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_NOISE",
            "timestamp": "2026-01-01T00:00:00",
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        }).json()

    # 3. Multivariate
    multi_idx = df[df['anomaly_type'] == 'multivariate'].index
    if len(multi_idx) > 0:
        row = df.loc[multi_idx[0]]
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_MULTI",
            "timestamp": "2026-01-01T00:00:00",
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        }).json()
