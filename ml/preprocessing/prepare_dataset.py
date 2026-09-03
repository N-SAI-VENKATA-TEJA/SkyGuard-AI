import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    print("======================================================")
    print("SkyGuard AI - Data Preprocessing (Step 2)")
    print("======================================================")
    
    # 1. Load raw data
    print(f"Loading raw data from: {config.RAW_DATA_PATH}")
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return
        
    original_rows = len(df)
    print(f"Raw rows: {original_rows}")
    
    # Extract only required columns
    required_cols = [config.COL_TIMESTAMP, config.COL_TEMPERATURE, config.COL_PRESSURE, config.COL_HUMIDITY]
    df = df[required_cols].copy()
    
    # 2. Parse timestamps
    print("\nParsing timestamps...")
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP], format="%d.%m.%Y %H:%M:%S")
    
    # 3. Handle Duplicates
    print("\nAnalyzing duplicates...")
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")
    
    timestamp_duplicates = df.duplicated(subset=[config.COL_TIMESTAMP]).sum()
    print(f"Duplicate timestamps total: {timestamp_duplicates}")
    
    conflicting_dups = timestamp_duplicates - exact_duplicates
    if conflicting_dups > 0:
        print(f"WARNING: Found {conflicting_dups} conflicting duplicate timestamps!")
        dups = df[df.duplicated(subset=[config.COL_TIMESTAMP], keep=False)]
        diff_check = dups.groupby(config.COL_TIMESTAMP).apply(lambda x: x.nunique().max() > 1)
        conflicts = diff_check[diff_check].index
        print(df[df[config.COL_TIMESTAMP].isin(conflicts)])
    else:
        print("No conflicting duplicates found. All duplicate timestamps are exact row duplicates.")
        
    # Safely remove exact duplicates
    if exact_duplicates > 0:
        print("Removing exact duplicate rows...")
        df = df.drop_duplicates()
        print(f"Rows after dropping exact duplicates: {len(df)}")
        
    # 4. Handle Out-of-Order Timestamps
    print("\nChecking timestamp order...")
    is_sorted = df[config.COL_TIMESTAMP].is_monotonic_increasing
    print(f"Data is chronologically sorted before sort: {is_sorted}")
    
    if not is_sorted:
        print("Sorting data chronologically...")
        df = df.sort_values(by=config.COL_TIMESTAMP)
        
    # Verify sorting
    assert df[config.COL_TIMESTAMP].is_monotonic_increasing, "Data is not monotonically increasing after sort!"
    print("Data is now strictly chronologically sorted.")
    
    # 5. Gap Analysis
    print("\nAnalyzing time gaps...")
    df = df.reset_index(drop=True)
    time_diffs = df[config.COL_TIMESTAMP].diff()
    print("Top 10 largest time gaps:")
    gaps = time_diffs.sort_values(ascending=False).head(10)
    for idx, gap in gaps.items():
        gap_start = df.loc[idx-1, config.COL_TIMESTAMP]
        gap_end = df.loc[idx, config.COL_TIMESTAMP]
        expected_missing = int(gap.total_seconds() // 600) - 1 # approx 10 min intervals
        print(f"Gap: {gap_start} to {gap_end} | Duration: {gap} | ~{expected_missing} missing 10-min obs")
        
    # Rename columns for processed dataset
    print("\nRenaming columns for clean processed dataset...")
    df = df.rename(columns={
        config.COL_TIMESTAMP: "timestamp",
        config.COL_TEMPERATURE: "temperature",
        config.COL_PRESSURE: "pressure",
        config.COL_HUMIDITY: "humidity"
    })
    
    # Save processed data
    out_path = os.path.join(config.PROCESSED_DATA_DIR, "aws_clean.csv")
    print(f"\nSaving processed dataset to {out_path}...")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed dataset saved successfully. Final shape: {df.shape}")
    print("\n======================================================")
    print("Preprocessing Complete.")
    print("======================================================")
    
if __name__ == "__main__":
    main()
