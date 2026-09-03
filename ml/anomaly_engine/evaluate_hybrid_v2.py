import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import seaborn as sns

def evaluate_models():
    # Load all predictions
    df_labels = pd.read_csv('data/processed/model_predictions.csv')
    df_v1 = pd.read_csv('data/processed/hybrid_predictions.csv')
    df_v2 = pd.read_csv('data/processed/hybrid_predictions_v2.csv')
    
    y_true = df_labels['is_anomaly']
    
    # 1. Overall Metrics
    def calc_metrics(y_pred, y_score=None):
        metrics = {
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0)
        }
        if y_score is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_score)
            metrics['PR-AUC'] = average_precision_score(y_true, y_score)
        return metrics

    res_stat = calc_metrics(df_labels['statistical_prediction'])
    res_pca = calc_metrics(df_labels['pca_prediction'])
    res_if = calc_metrics(df_labels['isolation_forest_prediction'])
    res_v1 = calc_metrics(df_v1['hybrid_prediction'], df_v1['hybrid_anomaly_score'])
    res_v2 = calc_metrics(df_v2['anomaly_flag'], df_v2['final_anomaly_score'])
    
    print("==================================================")
    print("OVERALL METRICS COMPARISON")
    print("==================================================")
    df_comp = pd.DataFrame({
        'Statistical': res_stat,
        'PCA': res_pca,
        'Isolation Forest': res_if,
        'Hybrid V1': res_v1,
        'Hybrid V2': res_v2
    }).T
    print(df_comp)
    
    # 2. Per Anomaly Type Metrics for V2
    df_v2['anomaly_type'] = df_labels['anomaly_type']
    types = df_v2['anomaly_type'].unique()
    types = [t for t in types if t != 'normal']
    
    print("\n==================================================")
    print("PER-TYPE RECALL (V2)")
    print("==================================================")
    for t in types:
        mask = df_v2['anomaly_type'] == t
        recall = df_v2.loc[mask, 'anomaly_flag'].mean()
        print(f"{t}: {recall*100:.2f}%")
        
    # 3. Event-level Recall
    df_v2['anomaly_id'] = df_labels['anomaly_id']
    df_v2['is_anomaly'] = df_labels['is_anomaly']
    events = df_v2[df_v2['is_anomaly'] == 1].groupby('anomaly_id')['anomaly_flag'].max()
    event_recall = events.mean()
    print(f"\nEvent-level Recall (V2): {event_recall*100:.2f}%")

    # 4. Plots
    os.makedirs('docs/validation', exist_ok=True)
    
    # Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_v2, x='final_anomaly_score', hue='anomaly_type', bins=50, multiple='stack')
    plt.title('Hybrid V2 Score Distribution by Anomaly Type')
    plt.savefig('docs/validation/hybrid_v2_score_distribution.png')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, df_v2['anomaly_flag'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Hybrid V2 Confusion Matrix')
    plt.savefig('docs/validation/hybrid_v2_confusion_matrix.png')
    plt.close()

    # Family scores
    plt.figure(figsize=(10, 6))
    df_anom = df_v2[df_v2['is_anomaly'] == 1]
    sns.scatterplot(data=df_anom, x='sudden_event_score', y='persistent_fault_score', hue='anomaly_type', alpha=0.6)
    plt.title('Hybrid V2: Sudden vs Persistent Scores (Anomalies Only)')
    plt.savefig('docs/validation/hybrid_v2_family_scores.png')
    plt.close()
    
    # Confidence vs Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_anom, x='final_anomaly_score', y='anomaly_confidence', hue='anomaly_type', alpha=0.6)
    plt.title('Hybrid V2: Confidence vs Final Score (Anomalies Only)')
    plt.savefig('docs/validation/hybrid_v2_confidence_vs_score.png')
    plt.close()
    
    # PR Curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    precision, recall, _ = precision_recall_curve(y_true, df_v2['final_anomaly_score'])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Hybrid V2 PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('docs/validation/hybrid_v2_pr_curve.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, df_v2['final_anomaly_score'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.title('Hybrid V2 ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('docs/validation/hybrid_v2_roc_curve.png')
    plt.close()
    
if __name__ == '__main__':
    evaluate_models()
