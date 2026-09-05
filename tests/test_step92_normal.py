import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api.main import app, pipelines, last_timestamps, station_locks

client = TestClient(app)

def reset_state():
    pipelines.clear()
    last_timestamps.clear()
    station_locks.clear()

def find_clean_segment(df, length):
    for start in range(len(df) - length):
        sub = df.iloc[start:start+length]
        if sub[['temperature', 'pressure', 'humidity']].isna().any().any():
            continue
        time_diffs = sub['timestamp'].diff().dt.total_seconds().dropna()
        if not (time_diffs == 600).all():
            continue
        return sub.copy().reset_index(drop=True)
    return None

def run_normal_experiment():
    df = pd.read_csv("data/processed/aws_clean.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    segment = find_clean_segment(df, 450)
    reset_state()
    
    print("\n--- RUNNING NORMAL DYNAMIC BASELINE ---")
    
    anomalies_flagged = 0
    for i, row in segment.iterrows():
        res = client.post("/api/v1/observations", json={
            "station_id": "AWS_NORMAL",
            "timestamp": row['timestamp'].isoformat(),
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        })
        
        data = res.json()
        if data['anomaly_flag'] and data['processing_state'] == 'PROCESSED':
            anomalies_flagged += 1
            print(f"False Positive at index {i}: Score {data['anomaly_score']}")
            
    if anomalies_flagged == 0:
        print("DETECTED 0 ANOMALIES (Normal sequence verified strictly clean)")
        
if __name__ == "__main__":
    run_normal_experiment()
