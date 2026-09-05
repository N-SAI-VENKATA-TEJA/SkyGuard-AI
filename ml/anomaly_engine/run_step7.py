import pandas as pd
import time
from ml.anomaly_engine.step7_sensor_health import SensorHealthTracker
import os

def run_step7():
    print("==================================================")
    print("STEP 7: FAULT CLASSIFICATION & SENSOR HEALTH ENGINE")
    print("==================================================")

    # 1. Load data
    print("Loading datasets...")
    df_hybrid = pd.read_csv('data/processed/hybrid_predictions_v2.csv')
    df_features = pd.read_csv('data/processed/aws_synthetic_features.csv')
    
    # 2. Merge necessary features
    feature_cols = [
        'timestamp',
        'temperature_rate_per_hour', 'pressure_rate_per_hour', 'humidity_rate_per_hour',
        'temperature_roll_z_60m', 'pressure_roll_z_60m', 'humidity_roll_z_60m'
    ]
    df_input = pd.merge(df_hybrid, df_features[feature_cols], on='timestamp', how='left')
    
    tracker = SensorHealthTracker()
    results = []
    
    print("Processing rows iteratively (stateful)...")
    start_time = time.time()
    
    # We can process iteratively or row by row using a dict
    rows = df_input.to_dict('records')
    for row in rows:
        out = tracker.process_row(row)
        results.append(out)
        
    end_time = time.time()
    elapsed = end_time - start_time
    rows_per_sec = len(rows) / elapsed if elapsed > 0 else 0
    
    print(f"Processed {len(rows)} rows in {elapsed:.2f} seconds ({rows_per_sec:.0f} rows/sec).")
    
    df_out = pd.DataFrame(results)
    output_path = 'data/processed/step7_sensor_health.csv'
    df_out.to_csv(output_path, index=False)
    print(f"Saved Step 7 outputs to {output_path}")

if __name__ == '__main__':
    run_step7()
