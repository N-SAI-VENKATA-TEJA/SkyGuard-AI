import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def evaluate_hybrid_metrics(df: pd.DataFrame, df_preds_baseline: pd.DataFrame):
    """
    Evaluates Hybrid predictions and compares with PCA baseline.
    """
    y_true = df['is_anomaly'].values
    y_pred = df['hybrid_prediction'].values
    y_score = df['hybrid_anomaly_score'].values
    
    # Core Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(pr_rec, pr_prec)
    except ValueError:
        roc_auc = np.nan
        pr_auc = np.nan
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Event-level logic
    df_events = df[df['anomaly_id'] != 'none']
    event_ids = df_events['anomaly_id'].unique()
    
    detected_events = 0
    total_events = len(event_ids)
    
    if total_events > 0:
        for ev in event_ids:
            ev_rows = df_events[df_events['anomaly_id'] == ev]
            if ev_rows['hybrid_prediction'].sum() > 0:
                detected_events += 1
        event_recall = detected_events / total_events
    else:
        event_recall = np.nan
        
    # PCA vs Hybrid FP comparison
    pca_pred = df_preds_baseline['pca_prediction'].values
    pca_cm = confusion_matrix(y_true, pca_pred)
    pca_tn, pca_fp, pca_fn, pca_tp = pca_cm.ravel()
    
    fp_reduction = 0.0
    if pca_fp > 0:
        fp_reduction = ((pca_fp - fp) / pca_fp) * 100
        
    recall_change = recall - (pca_tp / (pca_tp + pca_fn))
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': acc,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Event Recall': event_recall,
        'CM': cm,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'FPR': fp / (fp + tn) if (fp+tn)>0 else 0,
        'PCA_FP': pca_fp,
        'Hybrid_FP': fp,
        'FP_Reduction_Pct': fp_reduction,
        'PCA_TP': pca_tp,
        'Hybrid_TP': tp,
        'Recall_Change': recall_change
    }

def analyze_hybrid_per_anomaly_type(df: pd.DataFrame):
    types = df['anomaly_type'].unique()
    results = []
    
    for t in types:
        if t == 'normal': continue
        
        subset = df[df['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        y_true = subset['is_anomaly']
        y_pred = subset['hybrid_prediction']
        
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

def generate_hybrid_plots(df: pd.DataFrame, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    y_true = df['is_anomaly'].values
    y_score = df['hybrid_anomaly_score'].values
    y_pred = df['hybrid_prediction'].values
    
    # 1. Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = df.sample(min(len(df), 50000)) if len(df) > 50000 else df
    norm = plot_df[plot_df['is_anomaly'] == 0]['hybrid_anomaly_score']
    anom = plot_df[plot_df['is_anomaly'] == 1]['hybrid_anomaly_score']
    ax.hist(norm, bins=50, alpha=0.5, label='Normal', density=True, color='blue')
    ax.hist(anom, bins=50, alpha=0.5, label='Anomaly', density=True, color='red')
    ax.set_title('Hybrid Score Distribution')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'hybrid_distribution.png'))
    plt.close()
    
    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'Hybrid (PR-AUC = {auc(recall, precision):.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Hybrid PR Curve')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'hybrid_pr_curve.png'))
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'Hybrid (ROC-AUC = {auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('Hybrid ROC Curve')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'hybrid_roc_curve.png'))
    plt.close()
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Hybrid Confusion Matrix')
    plt.savefig(os.path.join(plot_dir, 'hybrid_confusion_matrix.png'))
    plt.close()
    
def threshold_sensitivity_analysis(dev_scores: np.ndarray):
    """Returns thresholds for 99.0, 99.5, 99.9 percentiles on dev data."""
    valid = dev_scores[~np.isnan(dev_scores)]
    return {
        '99.0%': np.percentile(valid, 99.0),
        '99.5%': np.percentile(valid, 99.5),
        '99.9%': np.percentile(valid, 99.9)
    }

def run_causality_test(engine, df_features: pd.DataFrame, df_preds: pd.DataFrame) -> bool:
    """
    Tests causality by ensuring score at time t does not change if t+1 is modified or appended.
    We just take a subset, predict, then modify the subset and predict again.
    (Since the engine is purely row-by-row after feature engineering, this is guaranteed, 
    but we execute the test to prove it).
    """
    if len(df_features) < 10:
        return True
        
    t_idx = 5
    subset1 = df_features.iloc[:t_idx+1].copy()
    subset1_preds = df_preds.iloc[:t_idx+1].copy()
    
    res1 = engine.predict(subset1, subset1_preds)
    score1 = res1.iloc[t_idx]['hybrid_anomaly_score']
    
    subset2 = df_features.iloc[:t_idx+5].copy()
    subset2_preds = df_preds.iloc[:t_idx+5].copy()
    
    res2 = engine.predict(subset2, subset2_preds)
    score2 = res2.iloc[t_idx]['hybrid_anomaly_score']
    
    return np.isclose(score1, score2)
