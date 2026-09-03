import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .feature_engineering import FeatureEngineer

def validate_features(df_feat: pd.DataFrame, df_orig: pd.DataFrame):
    """
    Validates feature dataset for row counts, label preservation, nans, and info.
    """
    print("\n--- Feature Validation ---")
    
    # 1. Row count & timestamp sequence
    assert len(df_feat) == len(df_orig), f"Row count mismatch! Feat: {len(df_feat)}, Orig: {len(df_orig)}"
    
    # Check timestamp sequence exactly matches
    assert (df_feat['timestamp'] == df_orig['timestamp']).all(), "Timestamp sequence mismatch!"
    print("[PASS] Row count and timestamps perfectly aligned.")

    # 2. Label Preservation
    labels = ['is_anomaly', 'anomaly_type', 'affected_sensor', 'anomaly_id']
    for l in labels:
        if l in df_orig.columns:
            assert (df_feat[l] == df_orig[l]).all(), f"Label '{l}' was modified or misaligned!"
    print("[PASS] All labels strictly preserved.")

    # 3. Missing Value Validation (Distinct Categories)
    print("\nMissing Value Analysis:")
    
    sensors = ['temperature', 'pressure', 'humidity']
    # A. Expected NaNs in raw sensors
    raw_nans = df_feat[sensors].isna().sum()
    print("A. Expected NaNs in raw sensors (Synthetic Missing events):")
    print(raw_nans[raw_nans > 0].to_dict() if raw_nans.sum() > 0 else "None")
    
    # Identify initial rows which are naturally NaN due to rolling windows
    # Max window is 360 mins. Roughly 36 rows at 10-min intervals.
    initial_idx = df_feat['timestamp'] < (df_feat['timestamp'].iloc[0] + pd.Timedelta(minutes=360))
    df_initial = df_feat[initial_idx]
    df_stable = df_feat[~initial_idx]
    
    # B. Expected initial NaNs (due to rolling)
    initial_nans = df_initial.isna().sum()
    print(f"B. Initial NaNs in first {len(df_initial)} rows (Rolling window warm-up):")
    # Only show ones that are actually NaN
    initial_nans_dict = initial_nans[initial_nans > 0].to_dict()
    print(f"   Found {len(initial_nans_dict)} features with initial NaNs.")

    # C. Unexpected NaNs (in the stable region)
    # Exclude raw sensors and known missing features
    feature_cols = [c for c in df_feat.columns if c not in labels + ['timestamp'] + sensors]
    unexpected_nans = df_stable[feature_cols].isna().sum()
    unexpected = unexpected_nans[unexpected_nans > 0]
    print("C. Unexpected NaNs in engineered features (after warm-up):")
    if len(unexpected) > 0:
        print(unexpected.to_dict())
    else:
        print("   None.")
        
    # D. Unexpected Infinite values
    # Replace inf with nan temporarily just to check sum
    has_inf = np.isinf(df_feat[feature_cols].select_dtypes(include=np.number)).sum()
    inf_cols = has_inf[has_inf > 0]
    print("D. Unexpected Infinite Values:")
    if len(inf_cols) > 0:
        print(inf_cols.to_dict())
    else:
        print("   None.")
        
    # 4. Constant and Near-Constant Features
    print("\nFeature Quality:")
    stds = df_feat[feature_cols].std(numeric_only=True)
    constant_features = stds[stds == 0].index.tolist()
    print(f"Constant features: {len(constant_features)} {constant_features}")
    
    print("--- Validation Complete ---\n")

def run_leakage_test(feature_engineer, df: pd.DataFrame):
    """
    Runs causal leakage test at multiple representative locations.
    """
    print("\n--- Running Leakage Test ---")
    
    # Identify interesting indices
    test_indices = []
    
    # 1. Normal period
    normal_idx = df[df['anomaly_type'] == 'normal'].index
    if len(normal_idx) > 100: test_indices.append(normal_idx[100])
    
    # 2. Spike
    spike_idx = df[df['anomaly_type'] == 'spike'].index
    if len(spike_idx) > 0: test_indices.append(spike_idx[0])
        
    # 3. Drift
    drift_idx = df[df['anomaly_type'] == 'drift'].index
    if len(drift_idx) > 0: test_indices.append(drift_idx[len(drift_idx)//2])
        
    # 4. Frozen
    frozen_idx = df[df['anomaly_type'] == 'frozen'].index
    if len(frozen_idx) > 0: test_indices.append(frozen_idx[len(frozen_idx)//2])
        
    # 5. Missing
    missing_idx = df[df['anomaly_type'] == 'missing'].index
    if len(missing_idx) > 0: test_indices.append(missing_idx[0])

    print(f"Testing {len(test_indices)} representative timestamps for leakage...")

    for idx in test_indices:
        t = df.iloc[idx]['timestamp']
        a_type = df.iloc[idx]['anomaly_type']
        
        # 1. Compute using data up to t
        df_past = df.iloc[:idx+1].copy()
        feat_past = feature_engineer.transform(df_past)
        row_past = feat_past.iloc[-1]
        
        # 2. Compute using all data (including future)
        # We simulate this by transforming a larger chunk of data extending past t
        df_all = df.iloc[:idx+100].copy() if idx+100 < len(df) else df.copy()
        feat_all = feature_engineer.transform(df_all)
        row_all = feat_all.iloc[idx] # Same timestamp
        
        # 3. Verify exact match for all features
        # Exclude labels as they are just copied
        ignore_cols = ['is_anomaly', 'anomaly_type', 'affected_sensor', 'anomaly_id', 'timestamp']
        check_cols = [c for c in feat_past.columns if c not in ignore_cols]
        
        for c in check_cols:
            v_past = row_past[c]
            v_all = row_all[c]
            
            # Handle NaNs safely
            if pd.isna(v_past) and pd.isna(v_all):
                continue
                
            assert np.isclose(v_past, v_all, equal_nan=True) or v_past == v_all, f"LEAKAGE DETECTED in '{c}' at index {idx} ({a_type})! Past: {v_past}, All: {v_all}"
            
    print("[PASS] Strict causal leakage test passed at all test locations. No future data contamination.")
    print("----------------------------\n")


def analyze_feature_by_anomaly(df_feat: pd.DataFrame):
    """
    Descriptive feature analysis grouped by anomaly type.
    """
    print("\n--- Feature vs Anomaly Analysis (Descriptive) ---")
    if 'anomaly_type' not in df_feat.columns:
        print("No anomaly_type found.")
        return
        
    # Select a few key features
    key_features = [
        'temperature_dev_mean_60m', 
        'pressure_rate_per_hour',
        'humidity_consec_unchanged',
        'multivariate_z_disagreement',
        'all_sensors_missing'
    ]
    
    # Filter features that exist
    features_to_analyze = [f for f in key_features if f in df_feat.columns]
    
    for feat in features_to_analyze:
        print(f"\nFeature: {feat}")
        grouped = df_feat.groupby('anomaly_type')[feat].agg(['count', 'median', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
        grouped.columns = ['Count', 'Median', '5th Pctl', '95th Pctl']
        print(grouped)
        
    print("-------------------------------------------------\n")


def generate_validation_plots(df_feat: pd.DataFrame, output_dir: str):
    """
    Generates validation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Raw temperature + temporal features around a spike
    spike_idx = df_feat[df_feat['anomaly_type'] == 'spike'].index
    if len(spike_idx) > 0:
        idx = spike_idx[0]
        window = df_feat.iloc[max(0, idx-20):min(len(df_feat), idx+20)]
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(window['timestamp'], window['temperature'], marker='o', label='Temperature (Raw)')
        ax1.set_ylabel('Temperature')
        ax2 = ax1.twinx()
        ax2.plot(window['timestamp'], window['temperature_abs_delta_1'], color='red', linestyle='--', label='Abs Delta 1')
        ax2.set_ylabel('Absolute Delta')
        
        plt.title('Spike Anomaly and First Order Delta')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.savefig(os.path.join(output_dir, '1_spike_delta.png'))
        plt.close()
        
    # Plot 2: Raw sensor + rolling baseline around a drift
    drift_idx = df_feat[df_feat['anomaly_type'] == 'drift'].index
    if len(drift_idx) > 0:
        idx = drift_idx[len(drift_idx)//2]
        window = df_feat.iloc[max(0, idx-50):min(len(df_feat), idx+20)]
        sensor = df_feat.iloc[idx]['affected_sensor']
        if sensor in ['temperature', 'pressure', 'humidity']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(window['timestamp'], window[sensor], label=f'{sensor.capitalize()} (Raw)')
            ax.plot(window['timestamp'], window[f'{sensor}_roll_mean_60m'], label='Rolling Mean (60m)', linestyle='--')
            plt.title(f'Drift Anomaly in {sensor.capitalize()} with Causal Baseline')
            plt.legend()
            plt.savefig(os.path.join(output_dir, '2_drift_baseline.png'))
            plt.close()

    # Plot 3: Frozen sensor + unchanged count
    frozen_idx = df_feat[df_feat['anomaly_type'] == 'frozen'].index
    if len(frozen_idx) > 0:
        idx = frozen_idx[len(frozen_idx)//2]
        window = df_feat.iloc[max(0, idx-30):min(len(df_feat), idx+30)]
        sensor = df_feat.iloc[idx]['affected_sensor']
        if sensor in ['temperature', 'pressure', 'humidity']:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(window['timestamp'], window[sensor], marker='.', label=f'{sensor.capitalize()} (Raw)')
            ax2 = ax1.twinx()
            ax2.plot(window['timestamp'], window[f'{sensor}_consec_unchanged'], color='orange', label='Consecutive Unchanged')
            plt.title(f'Frozen Anomaly in {sensor.capitalize()} and Unchanged Counter')
            fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
            plt.savefig(os.path.join(output_dir, '3_frozen_counter.png'))
            plt.close()
            
    # Plot 4: Missing data indicators
    missing_idx = df_feat[df_feat['anomaly_type'] == 'missing'].index
    if len(missing_idx) > 0:
        idx = missing_idx[0]
        window = df_feat.iloc[max(0, idx-10):min(len(df_feat), idx+20)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(window['timestamp'], window['temperature'], marker='o', label='Temperature')
        ax.plot(window['timestamp'], window['any_sensor_missing']*10, marker='x', label='Any Sensor Missing (Scaled)', color='red')
        plt.title('Missing Data Indicators')
        plt.legend()
        plt.savefig(os.path.join(output_dir, '4_missing_indicators.png'))
        plt.close()

    # Plot 5: Multivariate inconsistency
    multi_idx = df_feat[df_feat['anomaly_type'] == 'multivariate_inconsistency'].index
    if len(multi_idx) > 0:
        idx = multi_idx[len(multi_idx)//2]
        window = df_feat.iloc[max(0, idx-30):min(len(df_feat), idx+30)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(window['timestamp'], window['temperature_roll_z_60m'], label='Temp Z-Score (60m)')
        ax.plot(window['timestamp'], window['humidity_roll_z_60m'], label='Humidity Z-Score (60m)')
        ax.plot(window['timestamp'], window['multivariate_z_disagreement'], label='Multivariate Disagreement', linestyle='--', color='black', linewidth=2)
        plt.title('Multivariate Inconsistency and Z-Score Disagreement')
        plt.legend()
        plt.savefig(os.path.join(output_dir, '5_multivariate_disagreement.png'))
        plt.close()

    print(f"Validation plots saved to {output_dir}")
