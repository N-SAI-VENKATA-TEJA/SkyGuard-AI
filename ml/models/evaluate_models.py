import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

def evaluate_predictions(df: pd.DataFrame, model_name: str, prefix: str):
    """
    Evaluates predictions and returns metrics.
    Prefix should be something like 'pca' mapping to 'pca_prediction' and 'pca_score'.
    """
    y_true = df['is_anomaly'].values
    y_pred = df[f'{prefix}_prediction'].values
    y_score = df[f'{prefix}_score'].values
    
    # Core Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan
        pr_auc = np.nan
        
    cm = confusion_matrix(y_true, y_pred)
    
    # Event-level logic
    # An event is detected if ANY row matching anomaly_id (where anomaly_id != 'none') is flagged
    df_events = df[df['anomaly_id'] != 'none']
    event_ids = df_events['anomaly_id'].unique()
    
    detected_events = 0
    total_events = len(event_ids)
    
    if total_events > 0:
        for ev in event_ids:
            ev_rows = df_events[df_events['anomaly_id'] == ev]
            if ev_rows[f'{prefix}_prediction'].sum() > 0:
                detected_events += 1
        event_recall = detected_events / total_events
    else:
        event_recall = np.nan
        
    return {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': acc,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Event Recall': event_recall,
        'CM': cm
    }

def analyze_per_anomaly_type(df: pd.DataFrame, prefix: str):
    """Calculates per-anomaly-type metrics."""
    types = df['anomaly_type'].unique()
    results = []
    
    for t in types:
        if t == 'normal': continue
        
        subset = df[df['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        y_true = subset['is_anomaly']
        y_pred = subset[f'{prefix}_prediction']
        
        detected = y_pred.sum()
        total = len(subset)
        recall = detected / total if total > 0 else 0
        
        results.append({
            'Anomaly Type': t,
            'Total Rows': total,
            'Detected Rows': detected,
            'Recall': recall
        })
        
    return pd.DataFrame(results)

def generate_validation_plots(df: pd.DataFrame, metrics: dict, output_dir: str):
    """Generate requested plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Score distributions
    for prefix, name in [('statistical', 'Statistical Baseline'), 
                         ('pca', 'PCA Detector'), 
                         ('isolation_forest', 'Isolation Forest')]:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Subsample for plotting speed if huge
        plot_df = df.sample(min(len(df), 50000)) if len(df) > 50000 else df
        
        norm = plot_df[plot_df['is_anomaly'] == 0][f'{prefix}_score']
        anom = plot_df[plot_df['is_anomaly'] == 1][f'{prefix}_score']
        
        ax.hist(norm, bins=50, alpha=0.5, label='Normal', density=True, color='blue')
        ax.hist(anom, bins=50, alpha=0.5, label='Anomaly', density=True, color='red')
        
        ax.set_title(f'Score Distribution: {name}')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_distribution.png'))
        plt.close()

    # 2 & 3. Precision-Recall and ROC Curves (Optional, skip if computationally heavy or plot directly)
    # We will do a simple Time-Series plot instead as it's more informative for events
    
    # 5. Time-series example
    event_ids = df[df['anomaly_id'] != 'none']['anomaly_id'].unique()
    if len(event_ids) > 0:
        ev = event_ids[0]
        ev_idx = df[df['anomaly_id'] == ev].index
        start = max(0, ev_idx[0] - 50)
        end = min(len(df), ev_idx[-1] + 50)
        
        window = df.iloc[start:end]
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Plot raw sensor (assume affected_sensor points to the column, or default to temperature)
        sensor = df.loc[ev_idx[0], 'affected_sensor']
        if sensor not in ['temperature', 'pressure', 'humidity']: sensor = 'temperature'
        
        ax1.plot(pd.to_datetime(window['timestamp']), window[sensor], label=f'{sensor.capitalize()} (Raw)', color='black')
        ax1.axvspan(pd.to_datetime(df.loc[ev_idx[0], 'timestamp']), 
                    pd.to_datetime(df.loc[ev_idx[-1], 'timestamp']), 
                    color='red', alpha=0.2, label='True Anomaly')
        ax1.legend(loc='upper right')
        
        ax2.plot(pd.to_datetime(window['timestamp']), window['statistical_score'], label='Statistical Score')
        ax2.legend(loc='upper right')
        
        ax3.plot(pd.to_datetime(window['timestamp']), window['pca_score'], label='PCA Score', color='orange')
        ax3.legend(loc='upper right')
        
        ax4.plot(pd.to_datetime(window['timestamp']), window['isolation_forest_score'], label='Isolation Forest Score', color='green')
        ax4.legend(loc='upper right')
        
        plt.suptitle(f"Anomaly Detection Example: {ev}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_example.png'))
        plt.close()
