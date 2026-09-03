import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
# Using seaborn directly might conflict with some backend issues sometimes, but plt.subplots is fine.
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze():
    print("Loading processed data...")
    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "aws_clean.csv"))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    docs_dir = os.path.join(config.BASE_DIR, "docs", "plots")
    os.makedirs(docs_dir, exist_ok=True)

    # Task 6 - Constant runs
    print("\n--- Investigating Constant-Value Runs ---")
    variables = ['temperature', 'pressure', 'humidity']
    for var in variables:
        is_diff = df[var].diff() != 0
        groups = is_diff.cumsum()
        run_lengths = df.groupby(groups)[var].transform('size')
        
        runs = df.groupby(groups).agg(
            start_time=('timestamp', 'first'),
            end_time=('timestamp', 'last'),
            duration=('timestamp', lambda x: x.max() - x.min()),
            run_length=(var, 'size'),
            value=(var, 'first')
        )
        
        constant_runs = runs[runs['run_length'] > 1].sort_values(by='run_length', ascending=False)
        total_runs = len(constant_runs)
        longest_run = constant_runs['run_length'].max()
        
        print(f"\nVariable: {var}")
        print(f"Total constant runs (>1 obs): {total_runs}")
        print(f"Longest run length: {longest_run}")
        
        print("Top 5 longest runs:")
        print(constant_runs.head(5)[['start_time', 'end_time', 'duration', 'run_length', 'value']])
        
        if var == 'humidity' and longest_run >= 239:
            longest_rh_run = constant_runs.iloc[0]
            start_idx = df[df['timestamp'] == longest_rh_run['start_time']].index[0]
            end_idx = df[df['timestamp'] == longest_rh_run['end_time']].index[0]
            print("\nContext for longest RH run (showing temp and pressure during this time):")
            # show a little before and after
            context = df.iloc[max(0, start_idx-2):min(len(df), end_idx+3)]
            print(context[['timestamp', 'temperature', 'pressure', 'humidity']])
            
    # Task 7 - Temporal behaviour
    print("\n--- Generating Temporal Plots ---")
    # Full historical
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    sns.lineplot(data=df, x='timestamp', y='temperature', ax=axes[0], color='red', linewidth=0.5)
    axes[0].set_title('Historical Temperature')
    
    sns.lineplot(data=df, x='timestamp', y='pressure', ax=axes[1], color='green', linewidth=0.5)
    axes[1].set_title('Historical Pressure')
    
    sns.lineplot(data=df, x='timestamp', y='humidity', ax=axes[2], color='blue', linewidth=0.5)
    axes[2].set_title('Historical Humidity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "historical_series.png"))
    plt.close()
    
    # Short period (first 7 days)
    first_week = df[df['timestamp'] <= df['timestamp'].min() + pd.Timedelta(days=7)]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    sns.lineplot(data=first_week, x='timestamp', y='temperature', ax=axes[0], color='red', marker='o', markersize=2)
    axes[0].set_title('First 7 Days - Temperature')
    
    sns.lineplot(data=first_week, x='timestamp', y='pressure', ax=axes[1], color='green', marker='o', markersize=2)
    axes[1].set_title('First 7 Days - Pressure')
    
    sns.lineplot(data=first_week, x='timestamp', y='humidity', ax=axes[2], color='blue', marker='o', markersize=2)
    axes[2].set_title('First 7 Days - Humidity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "first_7_days.png"))
    plt.close()

    # Task 8 - Distribution and Relationships
    print("\n--- Generating Distributions and Relationships ---")
    # Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['temperature'], bins=50, kde=True, ax=axes[0], color='red')
    axes[0].set_title('Temperature Distribution')
    
    sns.histplot(df['pressure'], bins=50, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Pressure Distribution')
    
    sns.histplot(df['humidity'], bins=50, kde=True, ax=axes[2], color='blue')
    axes[2].set_title('Humidity Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "distributions.png"))
    plt.close()
    
    # Relationships
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # using scatter plot with alpha for dense data
    # Subsampling for plot clarity if data is too big, but scatter is ok
    plot_df = df.sample(min(10000, len(df)), random_state=42) # sample for faster scatter
    
    sns.scatterplot(data=plot_df, x='temperature', y='humidity', ax=axes[0], alpha=0.1, color='purple')
    axes[0].set_title('Temp vs Humidity')
    
    sns.scatterplot(data=plot_df, x='temperature', y='pressure', ax=axes[1], alpha=0.1, color='orange')
    axes[1].set_title('Temp vs Pressure')
    
    sns.scatterplot(data=plot_df, x='pressure', y='humidity', ax=axes[2], alpha=0.1, color='cyan')
    axes[2].set_title('Pressure vs Humidity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "relationships.png"))
    plt.close()
    
    # Correlations
    corr = df[['temperature', 'pressure', 'humidity']].corr()
    print("\nPearson Correlation Matrix:")
    print(corr)
    
    print("\nDone generating analysis and plots.")

if __name__ == "__main__":
    analyze()
