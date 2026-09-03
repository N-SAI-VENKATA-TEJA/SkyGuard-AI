import os
import sys
import pandas as pd
import numpy as np

# Add parent ml/ directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def audit_data():
    print("==================================================")
    print("SkyGuard AI - Data Audit Script")
    print("==================================================")
    
    # 1. Load data
    try:
        print(f"Loading data from: {config.RAW_DATA_PATH}...")
        df = pd.read_csv(config.RAW_DATA_PATH)
        print("Successfully loaded dataset.\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Check basics
    print("--- Basic Information ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    required_cols = [config.COL_TIMESTAMP, config.COL_TEMPERATURE, config.COL_PRESSURE, config.COL_HUMIDITY]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        return
        
    print("\nData Types:")
    print(df[required_cols].dtypes)
    
    # 3. Time analysis
    print("\n--- Temporal Analysis ---")
    # Parsing dates can be slow, using dayfirst based on standard Max Planck CSV
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP], format="%d.%m.%Y %H:%M:%S")
    
    print(f"Minimum Timestamp: {df[config.COL_TIMESTAMP].min()}")
    print(f"Maximum Timestamp: {df[config.COL_TIMESTAMP].max()}")
    
    time_diffs = df[config.COL_TIMESTAMP].diff().dropna()
    print("\nTimestamp Interval Statistics (Timedelta):")
    print(time_diffs.describe())
    
    print("\n--- Duplicates and Missing Values ---")
    duplicate_rows = df.duplicated().sum()
    duplicate_timestamps = df.duplicated(subset=[config.COL_TIMESTAMP]).sum()
    print(f"Duplicate Rows: {duplicate_rows}")
    print(f"Duplicate Timestamps: {duplicate_timestamps}")
    
    print("\nMissing values (Core columns):")
    print(df[required_cols].isnull().sum())
    
    # 4. Statistical Analysis
    print("\n--- Core Parameters Statistics ---")
    for col in required_cols[1:]:
        print(f"\n{col}:")
        print(f"  Min:  {df[col].min():.4f}")
        print(f"  Max:  {df[col].max():.4f}")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Std:  {df[col].std():.4f}")
        
    # 5. Physical Range Checks
    print("\n--- Physical Range Checks ---")
    temp_violations = ((df[config.COL_TEMPERATURE] < -50) | (df[config.COL_TEMPERATURE] > 60)).sum()
    pressure_violations = (df[config.COL_PRESSURE] <= 0).sum()
    rh_violations = ((df[config.COL_HUMIDITY] < 0) | (df[config.COL_HUMIDITY] > 100)).sum()
    print(f"Temperature Out of Range (<-50 or >60): {temp_violations}")
    print(f"Pressure Out of Range (<=0): {pressure_violations}")
    print(f"RH Out of Range (<0 or >100): {rh_violations}")
    
    # 6. Constant Run Analysis
    print("\n--- Constant Run Analysis ---")
    for col in required_cols[1:]:
        is_diff = df[col].diff() != 0
        groups = is_diff.cumsum()
        run_lengths = df.groupby(groups)[col].transform('size')
        
        constant_readings = (run_lengths > 1).sum()
        max_run = run_lengths.max()
        
        print(f"{col}:")
        print(f"  Total readings in a constant run: {constant_readings}")
        print(f"  Longest constant run length: {max_run}")

    print("\n==================================================")
    print("Audit Complete.")
    print("==================================================")

if __name__ == "__main__":
    audit_data()
