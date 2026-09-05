import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api.main import app, pipelines, last_timestamps, station_locks
from simulator.anomaly_injector import AnomalyInjector
from simulator.config import ANOMALY_PARAMS

client = TestClient(app)

def reset_state():
    pipelines.clear()
    last_timestamps.clear()
    station_locks.clear()

def find_clean_segment(df, length):
    # Find a contiguous segment of length without NaNs and with exact 10 min deltas
    for start in range(len(df) - length):
        sub = df.iloc[start:start+length]
        if sub[['temperature', 'pressure', 'humidity']].isna().any().any():
            continue
            
        time_diffs = sub['timestamp'].diff().dt.total_seconds().dropna()
        if not (time_diffs == 600).all():
            continue
            
        return sub.copy().reset_index(drop=True)
    return None

def run_experiment():
    df = pd.read_csv("data/processed/aws_clean.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # We need about 450 points. 400 for warmup, 50 for anomaly window
    segment = find_clean_segment(df, 450)
    if segment is None:
        print("COULD NOT FIND CLEAN SEGMENT")
        return
        
    print(f"Base segment found: {segment['timestamp'].iloc[0]} to {segment['timestamp'].iloc[-1]}")
    
    injector = AnomalyInjector(random_seed=42)
    
    experiments = [
        ('offset', 'temperature'),
        ('noise', 'temperature'),
        ('multivariate_inconsistency', 'pressure')
    ]
    
    for atype, sensor in experiments:
        print(f"\n--- RUNNING {atype.upper()} ---")
        reset_state()
        
        # Inject
        mod_segment = segment.copy()
        params = ANOMALY_PARAMS[atype]
        
        start_idx = 400
        duration = 10
        
        if atype == 'offset':
            mod_segment[sensor] = injector.inject_offset(mod_segment[sensor], start_idx, duration, 
                                                         params[sensor]['min_offset'], params[sensor]['max_offset'])
            mag_info = f"Offset applied"
        elif atype == 'noise':
            mod_segment[sensor] = injector.inject_noise(mod_segment[sensor], start_idx, duration, 
                                                        params[sensor]['std_dev'])
            mag_info = f"Noise (std {params[sensor]['std_dev']})"
        elif atype == 'multivariate_inconsistency':
            mod_segment = injector.inject_multivariate(mod_segment, sensor, start_idx, duration, 
                                                       params[sensor]['min_offset'], params[sensor]['max_offset'])
            mag_info = "Multivariate inconsistency applied"

        detected_at = None
        first_detected_result = None
        
        for i, row in mod_segment.iterrows():
            res = client.post("/api/v1/observations", json={
                "station_id": f"AWS_{atype.upper()}",
                "timestamp": row['timestamp'].isoformat(),
                "temperature": row['temperature'],
                "pressure": row['pressure'],
                "humidity": row['humidity']
            })
            
            data = res.json()
            
            # Check for detection only during or right after anomaly (idx 400 to 410)
            if i >= start_idx and i < start_idx + duration:
                if data['anomaly_flag'] and detected_at is None:
                    detected_at = i
                    first_detected_result = data
                    
        print(f"Anomaly Injected at idx {start_idx} (Duration {duration}) on {sensor}")
        print(f"Magnitude info: {mag_info}")
        if detected_at is not None:
            print(f"DETECTED! (At idx {detected_at})")
            print(f"Score: {first_detected_result['anomaly_score']}, Severity: {first_detected_result['severity']}")
            print(f"Type: {first_detected_result['fault_type']}, Sensor: {first_detected_result['affected_sensor']}")
            print(f"Explanation: {first_detected_result['explanation']}")
        else:
            print("NOT DETECTED (Detection Limitation)")

if __name__ == "__main__":
    run_experiment()
