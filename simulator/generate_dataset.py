import os
import sys
import pandas as pd
import numpy as np
import uuid
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ml import config as ml_config
from simulator import config as sim_config
from simulator.anomaly_injector import AnomalyInjector

def is_safe_window(df, start_idx, duration):
    """Check if the window has continuous 10-min sampling and no pre-existing NaNs."""
    end_idx = start_idx + duration - 1
    if end_idx >= len(df):
        return False
        
    time_diff = (df['timestamp'].iloc[end_idx] - df['timestamp'].iloc[start_idx]).total_seconds()
    expected_diff = (duration - 1) * 600 # 10 mins = 600 secs
    
    # Allow some tolerance for slight irregular sampling (e.g., 9-11 min gaps)
    # If the time span is larger than expected by more than say, 30 minutes, it's unsafe (gap exists).
    if abs(time_diff - expected_diff) > 1800:
        return False
        
    # Check for NaNs
    if df.iloc[start_idx:end_idx+1][['temperature', 'pressure', 'humidity']].isna().any().any():
        return False
        
    return True

def generate_dataset(output_filename, subset_size=None, anomaly_fraction=0.05, specific_type=None):
    print(f"Generating dataset: {output_filename}")
    
    # Load clean data
    input_path = os.path.join(ml_config.PROCESSED_DATA_DIR, "aws_clean.csv")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # If subset size is requested, take a random continuous slice
    if subset_size is not None and subset_size < len(df):
        max_start = len(df) - subset_size
        rng = np.random.default_rng(sim_config.RANDOM_SEED + 100)
        start = rng.integers(0, max_start)
        df = df.iloc[start:start+subset_size].copy().reset_index(drop=True)
        print(f"Using random continuous subset of {subset_size} rows.")
    
    injector = AnomalyInjector(random_seed=sim_config.RANDOM_SEED)
    rng = np.random.default_rng(sim_config.RANDOM_SEED)
    
    # Setup label columns
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'normal'
    df['affected_sensor'] = 'none'
    df['anomaly_id'] = 'none'
    
    labels = []
    
    anomaly_types = ['spike', 'drift', 'frozen', 'offset', 'noise', 'missing', 'multivariate_inconsistency']
    sensors = ['temperature', 'pressure', 'humidity']
    
    # Estimate total points we want to convert to anomalies
    target_anomaly_points = int(len(df) * anomaly_fraction)
    current_anomaly_points = 0
    
    attempts = 0
    max_attempts = 100000
    
    while current_anomaly_points < target_anomaly_points and attempts < max_attempts:
        attempts += 1
        
        # Select type
        atype = specific_type if specific_type else rng.choice(anomaly_types)
        params = sim_config.ANOMALY_PARAMS[atype]
        
        # Select duration
        if 'duration_obs' in params: # spike
            duration = params['duration_obs']
        else:
            duration = rng.integers(params['min_duration_obs'], params['max_duration_obs'] + 1)
            
        sensor = rng.choice(sensors)
        start_idx = rng.integers(0, len(df) - duration)
        
        # Verify safe
        if not is_safe_window(df, start_idx, duration):
            continue
            
        # Avoid overlapping anomalies
        if df['is_anomaly'].iloc[start_idx:start_idx+duration].any():
            continue
            
        anomaly_id = str(uuid.uuid4())[:8]
        
        # Inject
        if atype == 'spike':
            df[sensor] = injector.inject_spike(df[sensor], start_idx, duration, params[sensor]['min_mag'], params[sensor]['max_mag'])
        elif atype == 'drift':
            df[sensor] = injector.inject_drift(df[sensor], start_idx, duration, params[sensor]['min_total_drift'], params[sensor]['max_total_drift'])
        elif atype == 'frozen':
            df[sensor] = injector.inject_frozen(df[sensor], start_idx, duration)
        elif atype == 'offset':
            df[sensor] = injector.inject_offset(df[sensor], start_idx, duration, params[sensor]['min_offset'], params[sensor]['max_offset'])
        elif atype == 'noise':
            df[sensor] = injector.inject_noise(df[sensor], start_idx, duration, params[sensor]['std_dev'])
        elif atype == 'missing':
            # all sensors go missing or just one? Usually AWS failure takes out all, but let's apply to all for communication failure.
            # actually prompt says "affect observations accordingly". We'll do all sensors for 'missing'
            df['temperature'] = injector.inject_missing(df['temperature'], start_idx, duration)
            df['pressure'] = injector.inject_missing(df['pressure'], start_idx, duration)
            df['humidity'] = injector.inject_missing(df['humidity'], start_idx, duration)
            sensor = 'all'
        elif atype == 'multivariate_inconsistency':
            df = injector.inject_multivariate(df, sensor, start_idx, duration, params[sensor]['min_offset'], params[sensor]['max_offset'])
        
        # Labeling
        df.loc[start_idx:start_idx+duration-1, 'is_anomaly'] = 1
        df.loc[start_idx:start_idx+duration-1, 'anomaly_type'] = atype
        df.loc[start_idx:start_idx+duration-1, 'affected_sensor'] = sensor
        df.loc[start_idx:start_idx+duration-1, 'anomaly_id'] = anomaly_id
        
        labels.append({
            'anomaly_id': anomaly_id,
            'start_timestamp': df['timestamp'].iloc[start_idx],
            'end_timestamp': df['timestamp'].iloc[start_idx+duration-1],
            'anomaly_type': atype,
            'affected_sensor': sensor,
            'severity': 'moderate',
            'duration_obs': duration
        })
        
        current_anomaly_points += duration

    # Save outputs
    out_dir = ml_config.PROCESSED_DATA_DIR
    df.to_csv(os.path.join(out_dir, output_filename), index=False)
    
    # Save labels if full dataset (for dev dataset we just append or use separate name)
    label_filename = f"labels_{output_filename}"
    pd.DataFrame(labels).to_csv(os.path.join(out_dir, label_filename), index=False)
    
    print(f"Injection complete. Anomalies injected: {len(labels)}")
    print(f"Target points: {target_anomaly_points}, Actual points: {current_anomaly_points}")
    print(f"Saved dataset to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Generate small dev dataset")
    args = parser.parse_args()
    
    if args.dev:
        generate_dataset("aws_dev_synthetic.csv", subset_size=15000, anomaly_fraction=0.08)
    else:
        generate_dataset("aws_synthetic_anomalies.csv", subset_size=None, anomaly_fraction=0.05)
