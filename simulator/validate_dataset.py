import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ml import config as ml_config

def validate_dataset(filename, is_dev=False):
    print(f"\n========================================================")
    print(f"Validating dataset: {filename}")
    print(f"========================================================")
    
    path = os.path.join(ml_config.PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    label_path = os.path.join(ml_config.PROCESSED_DATA_DIR, f"labels_{filename}")
    labels = pd.read_csv(label_path)
    
    print("\n--- DATASET INTEGRITY ---")
    print(f"Row count: {len(df)}")
    print(f"Timestamp monotonically increasing: {df['timestamp'].is_monotonic_increasing}")
    
    # Missing values (should only be from 'missing' anomaly type)
    missing_points = df[['temperature', 'pressure', 'humidity']].isna().sum().sum()
    print(f"Total missing value cells: {missing_points}")
    
    print("\n--- PHYSICAL RANGES ---")
    print(df[['temperature', 'pressure', 'humidity']].describe())
    
    print("\n--- INJECTION INTEGRITY ---")
    print(f"Total anomalies injected (from labels): {len(labels)}")
    print("Checking label bounds...")
    all_matched = True
    for _, row in labels.iterrows():
        start_ts = pd.to_datetime(row['start_timestamp'])
        end_ts = pd.to_datetime(row['end_timestamp'])
        mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
        slice_df = df[mask]
        
        if not (slice_df['is_anomaly'] == 1).all():
            print(f"ERROR: Label {row['anomaly_id']} has 'is_anomaly' != 1 in some rows")
            all_matched = False
        if not (slice_df['anomaly_type'] == row['anomaly_type']).all():
            print(f"ERROR: Label {row['anomaly_id']} mismatch in anomaly_type")
            all_matched = False
            
    print(f"Row-level labels perfectly align with injected intervals: {all_matched}")
    
    print("\n--- DISTRIBUTION ---")
    total_obs = len(df)
    anom_obs = df['is_anomaly'].sum()
    normal_obs = total_obs - anom_obs
    print(f"Total observations: {total_obs}")
    print(f"Normal observations: {normal_obs}")
    print(f"Anomalous observations: {anom_obs}")
    print(f"Anomaly percentage: {(anom_obs / total_obs) * 100:.2f}%")
    
    print("\nAnomaly distribution by type (observations):")
    print(df[df['is_anomaly'] == 1]['anomaly_type'].value_counts())
    
    print("\nAnomaly distribution by sensor (observations):")
    print(df[df['is_anomaly'] == 1]['affected_sensor'].value_counts())

    # Generate plots for dev dataset to keep it fast
    if is_dev:
        print("\n--- VISUAL VALIDATION ---")
        docs_dir = os.path.join(BASE_DIR, "docs", "validation")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Pick one example of each type
        types_to_plot = ['spike', 'drift', 'frozen', 'offset', 'missing', 'multivariate_inconsistency']
        for t in types_to_plot:
            type_labels = labels[labels['anomaly_type'] == t]
            if len(type_labels) == 0:
                continue
            example = type_labels.iloc[0]
            start_idx = df[df['timestamp'] == example['start_timestamp']].index[0]
            end_idx = df[df['timestamp'] == example['end_timestamp']].index[0]
            
            # window 50 before and 50 after
            win_start = max(0, start_idx - 50)
            win_end = min(len(df), end_idx + 50)
            
            win_df = df.iloc[win_start:win_end]
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            for i, var in enumerate(['temperature', 'pressure', 'humidity']):
                sns.lineplot(data=win_df, x='timestamp', y=var, ax=axes[i], color='blue')
                
                # Highlight anomalous region
                anom_region = win_df[win_df['anomaly_id'] == example['anomaly_id']]
                if not anom_region.empty:
                    axes[i].scatter(anom_region['timestamp'], anom_region[var], color='red', zorder=5)
                    
                axes[i].set_title(var)
                
            plt.suptitle(f"Anomaly Example: {t} (Sensor: {example['affected_sensor']})")
            plt.tight_layout()
            plt.savefig(os.path.join(docs_dir, f"anomaly_{t}.png"))
            plt.close()
            print(f"Saved plot for {t}")

if __name__ == "__main__":
    validate_dataset("aws_dev_synthetic.csv", is_dev=True)
    validate_dataset("aws_synthetic_anomalies.csv", is_dev=False)
