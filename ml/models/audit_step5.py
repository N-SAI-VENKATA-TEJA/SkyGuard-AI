import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.statistical_detector import StatisticalBaseline
from ml.models.pca_detector import PCADetector
from ml.models.isolation_forest_detector import IsolationForestDetector

def run_audit():
    print("==================================================")
    print("STEP 5 FINAL AUDIT")
    print("==================================================")
    
    dev_file = 'data/processed/aws_dev_features.csv'
    pred_file = 'data/processed/model_predictions.csv'
    plot_dir = 'docs/validation/models/'
    
    df_dev = pd.read_csv(dev_file)
    df_pred = pd.read_csv(pred_file)
    
    # ---------------------------------------------------------
    # 8. DATA INTEGRITY
    # ---------------------------------------------------------
    print("\n[DATA INTEGRITY VERIFICATION]")
    print(f"Row count: {len(df_pred)} (Expected: 420224)")
    print(f"Duplicate timestamps: {df_pred['timestamp'].duplicated().sum()}")
    print(f"Is chronological: {df_pred['timestamp'].is_monotonic_increasing}")
    print(f"NaNs in predictions: {df_pred[['statistical_prediction', 'pca_prediction', 'isolation_forest_prediction']].isna().sum().sum()}")
    print(f"Infinities in predictions: {np.isinf(df_pred[['statistical_score', 'pca_score', 'isolation_forest_score']]).sum().sum()}")
    
    # ---------------------------------------------------------
    # 1. VERIFY FLAG COUNTS & 7. METRIC CONSISTENCY
    # ---------------------------------------------------------
    print("\n[PER-MODEL FLAG COUNTS & METRICS]")
    y_true = df_pred['is_anomaly'].values
    total_eval = len(df_pred)
    
    for prefix, name in [('statistical', 'Statistical'), ('pca', 'PCA'), ('isolation_forest', 'Isolation Forest')]:
        y_pred = df_pred[f'{prefix}_prediction'].values
        y_score = df_pred[f'{prefix}_score'].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        predicted_anomalies = tp + fp
        predicted_normal = tn + fn
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"\nModel: {name}")
        print(f"  Total eval rows: {total_eval}")
        print(f"  Predicted Anomaly: {predicted_anomalies}")
        print(f"  Predicted Normal: {predicted_normal}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  False Positive Rate: {fpr:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
    # ---------------------------------------------------------
    # 2. VERIFY THE 134,772 FALSE-POSITIVE CLAIM
    # ---------------------------------------------------------
    print("\n[UNION FLAG ANALYSIS]")
    union_flag = (df_pred['statistical_prediction'] == 1) | \
                 (df_pred['pca_prediction'] == 1) | \
                 (df_pred['isolation_forest_prediction'] == 1)
                 
    union_pred = union_flag.astype(int).values
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_true, union_pred).ravel()
    
    print(f"Total rows in union_flag: {tp_u + fp_u}")
    print(f"Normal rows in union_flag (Union FP): {fp_u}")
    print(f"Anomalous rows in union_flag (Union TP): {tp_u}")
    
    # ---------------------------------------------------------
    # 3. REPORT DEVELOPMENT THRESHOLDS & 5. FLAG PERCENTAGES
    # ---------------------------------------------------------
    print("\n[DEVELOPMENT THRESHOLDS & DISTRIBUTIONS]")
    
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    
    stat_model.fit(df_dev, percentile=99.5)
    pca_model.fit(df_dev, percentile=99.5)
    if_model.fit(df_dev, percentile=99.5)
    
    s_score_dev, s_pred_dev = stat_model.predict(df_dev)
    p_score_dev, p_pred_dev = pca_model.predict(df_dev)
    i_score_dev, i_pred_dev = if_model.predict(df_dev)
    
    models = {
        'Statistical': (stat_model.threshold, s_score_dev, s_pred_dev, df_pred['statistical_prediction']),
        'PCA': (pca_model.threshold, p_score_dev, p_pred_dev, df_pred['pca_prediction']),
        'Isolation Forest': (if_model.threshold, i_score_dev, i_pred_dev, df_pred['isolation_forest_prediction'])
    }
    
    for name, (thresh, dev_scores, dev_preds, eval_preds) in models.items():
        print(f"\nModel: {name}")
        print(f"  Threshold: {thresh:.4f}")
        print(f"  Min dev score: {np.nanmin(dev_scores):.4f}")
        print(f"  Median dev score: {np.nanmedian(dev_scores):.4f}")
        print(f"  95th pctl dev score: {np.nanpercentile(dev_scores, 95):.4f}")
        print(f"  99th pctl dev score: {np.nanpercentile(dev_scores, 99):.4f}")
        print(f"  99.5th pctl dev score: {np.nanpercentile(dev_scores, 99.5):.4f}")
        print(f"  Max dev score: {np.nanmax(dev_scores):.4f}")
        
        dev_flag_pct = (dev_preds.sum() / len(dev_preds)) * 100
        eval_flag_pct = (eval_preds.sum() / len(eval_preds)) * 100
        
        print(f"  Dev Flag Percentage: {dev_flag_pct:.2f}%")
        print(f"  Eval Flag Percentage: {eval_flag_pct:.2f}%")

    # ---------------------------------------------------------
    # 6. RESTORE BENCHMARK VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n[GENERATING RESTORED PLOTS]")
    # Plot PR and ROC curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    
    for prefix, name, color in [('statistical', 'Statistical', 'blue'), 
                                ('pca', 'PCA', 'orange'), 
                                ('isolation_forest', 'Isolation Forest', 'green')]:
                                
        y_score = df_pred[f'{prefix}_score'].values
        y_score = np.nan_to_num(y_score, nan=0.0) # Handle nan if any
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})', color=color)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{name} (ROC-AUC = {roc_auc:.3f})', color=color)
        
        # Confusion Matrix
        y_pred = df_pred[f'{prefix}_prediction'].values
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.savefig(os.path.join(plot_dir, f'{prefix}_confusion_matrix.png'))
        plt.close()
        
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    fig_pr.savefig(os.path.join(plot_dir, 'combined_pr_curve.png'))
    plt.close(fig_pr)
    
    ax_roc.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    fig_roc.savefig(os.path.join(plot_dir, 'combined_roc_curve.png'))
    plt.close(fig_roc)
    
    print(f"Plots saved to {plot_dir}")
    print("\n--- AUDIT COMPLETE ---")

if __name__ == '__main__':
    run_audit()
