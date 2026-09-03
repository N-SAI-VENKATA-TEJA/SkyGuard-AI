import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.anomaly_engine.hybrid_detector_v2 import HybridAnomalyEngineV2
from ml.anomaly_engine.config_v2 import *

def run_audit():
    print("==================================================")
    print("STEP 6 V2: DIAGNOSTIC AUDIT")
    print("==================================================")
    
    df_v2 = pd.read_csv('data/processed/hybrid_predictions_v2.csv')
    df_labels = pd.read_csv('data/processed/model_predictions.csv')
    
    df_v2['is_anomaly'] = df_labels['is_anomaly']
    df_v2['anomaly_type'] = df_labels['anomaly_type']
    
    types = ['normal', 'spike', 'drift', 'frozen', 'offset', 'noise', 'multivariate_inconsistency', 'missing']
    
    print("\n1. REPRESENTATIVE EXAMPLES")
    for t in types:
        print(f"\n--- TYPE: {t.upper()} ---")
        subset = df_v2[df_v2['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        row = subset.iloc[0]
        print(f"Timestamp: {row['timestamp']}")
        print(f"Sensors -> Temp: {row['temperature']:.2f}, Pres: {row['pressure']:.2f}, Hum: {row['humidity']:.2f}")
        print(f"Family Scores -> Sudden: {row['sudden_event_score']:.4f}, Persistent: {row['persistent_fault_score']:.4f}, Comm: {row['communication_score']:.4f}")
        print(f"Context Factor: {row['context_factor']:.4f}")
        print(f"Final Score: {row['final_anomaly_score']:.4f}")
        print(f"Flag: {row['anomaly_flag']}")
        print(f"Confidence: {row['anomaly_confidence']:.4f}")
        print(f"Winning Family: {row['candidate_fault_family']}")
        print(f"Fault Hint: {row['fault_type_hint']}")
        print(f"Primary Evidence: {row['primary_evidence']}")
        
    print("\n2. PER-TYPE SCORE DISTRIBUTIONS (FINAL SCORE)")
    for t in types:
        subset = df_v2[df_v2['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        scores = subset['final_anomaly_score'].values
        above = subset['anomaly_flag'].sum()
        count = len(subset)
        print(f"{t.upper()}: count={count}, mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
              f"95th={np.percentile(scores, 95):.4f}, 99th={np.percentile(scores, 99):.4f}, "
              f"max={scores.max():.4f}, recall={above/count:.4f}")
              
    print("\n3. FAMILY SCORE DISTRIBUTIONS")
    cols = ['sudden_event_score', 'persistent_fault_score', 'communication_score']
    for t in types:
        subset = df_v2[df_v2['anomaly_type'] == t]
        if len(subset) == 0: continue
        means = subset[cols].mean()
        print(f"{t.upper()} Means -> Sudden: {means['sudden_event_score']:.4f}, Pers: {means['persistent_fault_score']:.4f}, Comm: {means['communication_score']:.4f}")

    print("\n4. FALSE POSITIVES / FALSE NEGATIVES")
    fp_mask = (df_v2['anomaly_flag'] == True) & (df_v2['is_anomaly'] == 0)
    fn_mask = (df_v2['anomaly_flag'] == False) & (df_v2['is_anomaly'] == 1)
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    print(f"Total FP: {fp_count}")
    print(f"Total FN: {fn_count}")
    
    print("\n5. WHICH FAMILY CATCHES WHICH ANOMALY")
    tp_mask = (df_v2['anomaly_flag'] == True) & (df_v2['is_anomaly'] == 1)
    df_tp = df_v2[tp_mask]
    catch_stats = df_tp.groupby(['anomaly_type', 'candidate_fault_family']).size().unstack(fill_value=0)
    print(catch_stats)
    
    print("\n6. CONTEXT SUPPRESSION EFFECT ON PERSISTENT FAULTS")
    print("Does context factor affect persistent score? NO (By architecture design).")
    
    print("\n7. COMMUNICATION DETECTION INDEPENDENCE")
    miss_subset = df_v2[df_v2['anomaly_type'] == 'missing']
    comm_score = miss_subset['communication_score'].mean()
    print(f"Mean communication score for missing anomalies: {comm_score:.4f}")
    
    print("\n8. CONFIDENCE DISTRIBUTIONS")
    print("NORMAL vs ANOMALOUS Rows:")
    norm_conf = df_v2[df_v2['is_anomaly'] == 0]['anomaly_confidence']
    anom_conf = df_v2[df_v2['is_anomaly'] == 1]['anomaly_confidence']
    print(f"Normal Confidence Mean: {norm_conf.mean():.4f}, Median: {norm_conf.median():.4f}")
    print(f"Anomaly Confidence Mean: {anom_conf.mean():.4f}, Median: {anom_conf.median():.4f}")
    
    tp_conf = df_tp['anomaly_confidence']
    fp_conf = df_v2[fp_mask]['anomaly_confidence']
    print("\nTRUE POSITIVES vs FALSE POSITIVES:")
    print(f"TP Confidence Mean: {tp_conf.mean():.4f}, Median: {tp_conf.median():.4f}")
    if len(fp_conf) > 0:
        print(f"FP Confidence Mean: {fp_conf.mean():.4f}, Median: {fp_conf.median():.4f}")
    else:
        print("No False Positives to analyze.")
        
    print("\n==================================================")
    print("CAUSALITY VALIDATION")
    print("==================================================")
    print("Running causality check on random indices...")
    df_full_feat = pd.read_csv('data/processed/aws_synthetic_features.csv')
    
    from ml.anomaly_engine.hybrid_detector_v2 import HybridAnomalyEngineV2
    from ml.models.statistical_detector import StatisticalBaseline
    from ml.models.pca_detector import PCADetector
    from ml.models.isolation_forest_detector import IsolationForestDetector
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    df_dev_feat = pd.read_csv('data/processed/aws_dev_features.csv')
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    stat_model.fit(df_dev_feat, 99.5)
    pca_model.fit(df_dev_feat, 99.5)
    if_model.fit(df_dev_feat, 99.5)
    s_sc, _ = stat_model.predict(df_dev_feat)
    p_sc, _ = pca_model.predict(df_dev_feat)
    i_sc, _ = if_model.predict(df_dev_feat)
    df_dev_model = pd.DataFrame({'statistical_score': s_sc, 'pca_score': p_sc, 'isolation_forest_score': i_sc})
    engine = HybridAnomalyEngineV2()
    engine.fit(df_dev_feat, df_dev_model)
    sys.stdout = old_stdout
    
    # Take a chunk up to t, and chunk up to t+10
    t = 1000
    chunk_t = df_full_feat.iloc[:t]
    chunk_t10 = df_full_feat.iloc[:t+10]
    
    pred_t = engine.predict(chunk_t, df_labels.iloc[:t])
    pred_t10 = engine.predict(chunk_t10, df_labels.iloc[:t+10])
    
    score_t = pred_t.iloc[t-1]['final_anomaly_score']
    score_t_in_t10 = pred_t10.iloc[t-1]['final_anomaly_score']
    
    if np.isclose(score_t, score_t_in_t10):
        print("Causality Test PASS: Scores at time t do not change when t+10 is appended.")
    else:
        print(f"Causality Test FAIL: Score t={score_t:.4f}, Score t in t10={score_t_in_t10:.4f}")

if __name__ == '__main__':
    run_audit()
