import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def verify():
    print("--- 1. PROCESSED DATASET VERIFICATION ---")
    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "aws_clean.csv"))
    print(f"Row count: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Min timestamp: {df['timestamp'].min()}")
    print(f"Max timestamp: {df['timestamp'].max()}")
    
    is_monotonic = df['timestamp'].is_monotonic_increasing
    print(f"Monotonically increasing: {is_monotonic}")
    
    time_diffs = df['timestamp'].diff().dropna()
    print(f"Negative diffs: {(time_diffs.dt.total_seconds() < 0).sum()}")
    print(f"Zero diffs: {(time_diffs.dt.total_seconds() == 0).sum()}")
    print(f"Duplicate timestamps: {df.duplicated(subset=['timestamp']).sum()}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"NaN values: {df.isna().sum().sum()}")
    print(f"Infinite values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")
    
    print("\n--- 2. LONGEST CONSTANT RUNS ---")
    for var in ['temperature', 'pressure', 'humidity']:
        is_diff = df[var].diff() != 0
        groups = is_diff.cumsum()
        runs = df.groupby(groups).agg(
            run_length=(var, 'size'),
            start_timestamp=('timestamp', 'first'),
            end_timestamp=('timestamp', 'last'),
            value=(var, 'first')
        ).sort_values('run_length', ascending=False)
        longest = runs.iloc[0]
        duration = longest['end_timestamp'] - longest['start_timestamp']
        print(f"\n{var.upper()}:")
        print(f"  Length: {longest['run_length']}")
        print(f"  Start: {longest['start_timestamp']}")
        print(f"  End: {longest['end_timestamp']}")
        print(f"  Duration: {duration}")
        print(f"  Value: {longest['value']}")

    print("\n--- 3. DUPLICATE VERIFICATION (RAW DATA) ---")
    raw_df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Raw rows: {len(raw_df)}")
    exact_dups = raw_df.duplicated().sum()
    ts_col = config.COL_TIMESTAMP
    ts_dups = raw_df.duplicated(subset=[ts_col]).sum()
    
    conflicting = ts_dups - exact_dups
    print(f"Exact duplicate rows: {exact_dups}")
    print(f"Duplicate timestamps: {ts_dups}")
    print(f"Conflicting duplicate timestamps: {conflicting}")
    print(f"Raw rows - exact duplicates == processed rows: {len(raw_df) - exact_dups == len(df)}")
    
    print("\n--- 5. GAP VERIFICATION ---")
    print(f"Median interval: {time_diffs.median()}")
    print(f"Most common interval: {time_diffs.mode()[0]}")
    print(f"Minimum interval: {time_diffs.min()}")
    print(f"Maximum interval: {time_diffs.max()}")
    print("Top 10 gaps:")
    gaps = time_diffs.sort_values(ascending=False).head(10)
    for idx, gap in gaps.items():
        gap_start = df.loc[idx-1, 'timestamp']
        gap_end = df.loc[idx, 'timestamp']
        print(f"  {gap_start} -> {gap_end} | Duration: {gap}")
        
    print("\n--- 6. PHYSICAL VALIDATION ---")
    print(f"Temp min: {df['temperature'].min()}, max: {df['temperature'].max()}, <-50: {(df['temperature'] < -50).sum()}, >60: {(df['temperature'] > 60).sum()}")
    print(f"Press min: {df['pressure'].min()}, max: {df['pressure'].max()}, <=0: {(df['pressure'] <= 0).sum()}")
    print(f"Hum min: {df['humidity'].min()}, max: {df['humidity'].max()}, <0: {(df['humidity'] < 0).sum()}, >100: {(df['humidity'] > 100).sum()}")
    
    print("\n--- 7. CORRELATION VERIFICATION ---")
    print(f"T-P: {df['temperature'].corr(df['pressure'])}")
    print(f"T-RH: {df['temperature'].corr(df['humidity'])}")
    print(f"P-RH: {df['pressure'].corr(df['humidity'])}")

if __name__ == "__main__":
    verify()
