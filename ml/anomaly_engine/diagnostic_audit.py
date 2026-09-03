import pandas as pd
import numpy as np
import inspect
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.anomaly_engine.hybrid_detector import HybridAnomalyEngine
from ml.anomaly_engine.config import FUSION_WEIGHTS, SUPPRESSION_FACTOR, COHERENT_MULTIVARIATE_THRESHOLD

def run_audit():
    print("==================================================")
    print("STEP 6 DIAGNOSTIC AUDIT PREPARATION")
    print("==================================================")
    
    hybrid_file = 'data/processed/hybrid_predictions.csv'
    model_file = 'data/processed/model_predictions.csv'
    
    df_hybrid = pd.read_csv(hybrid_file)
    df_model = pd.read_csv(model_file)
    
    # Add labels from model_predictions to df_hybrid since they were not saved in the hybrid output file
    df_hybrid['anomaly_type'] = df_model['anomaly_type']
    df_hybrid['is_anomaly'] = df_model['is_anomaly']
    
    # ---------------------------------------------------------
    # 1. REPRESENTATIVE CASE ANALYSIS
    # ---------------------------------------------------------
    print("\n[1. REPRESENTATIVE CASE ANALYSIS]")
    types = ['normal', 'spike', 'drift', 'frozen', 'offset', 'noise', 'multivariate_inconsistency', 'missing']
    
    for t in types:
        print(f"\n--- TYPE: {t.upper()} ---")
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        row = subset.iloc[0] # Pick first instance
        idx = row.name
        
        print(f"Timestamp: {row['timestamp']}")
        print(f"Sensors -> Temp: {row['temperature']:.2f}, Pres: {row['pressure']:.2f}, Hum: {row['humidity']:.2f}")
        
        print(f"temporal_evidence: {row['temporal_evidence']:.4f}")
        print(f"statistical_evidence: {row['statistical_evidence']:.4f}")
        print(f"multivariate_evidence: {row['multivariate_evidence']:.4f}")
        print(f"stability_evidence: {row['stability_evidence']:.4f}")
        print(f"missing_evidence: {row['missing_evidence']:.4f}")
        print(f"model_evidence: {row['model_evidence']:.4f}")
        
        print(f"contextual_consistency: {row['contextual_consistency']:.4f}")
        print(f"base_fusion_score: {row['base_score']:.4f}")
        print(f"suppression_factor: {row['suppression_factor']:.4f}")
        
        print(f"hybrid_anomaly_score: {row['hybrid_anomaly_score']:.4f}")
        print(f"hybrid_threshold: 0.9601")
        print(f"hybrid_prediction: {row['hybrid_prediction']}")
        
        # Step 5 scores for this row
        m_row = df_model.iloc[idx]
        print(f"statistical_score (Step 5): {m_row['statistical_score']:.4f}")
        print(f"pca_score (Step 5): {m_row['pca_score']:.4f}")
        print(f"isolation_forest_score (Step 5): {m_row['isolation_forest_score']:.4f}")
        
    # ---------------------------------------------------------
    # 2. PER-TYPE SCORE DISTRIBUTION
    # ---------------------------------------------------------
    print("\n[2. PER-TYPE SCORE DISTRIBUTION]")
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        scores = subset['hybrid_anomaly_score'].values
        above = subset['hybrid_prediction'].sum()
        count = len(subset)
        print(f"{t.upper()}: count={count}, mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
              f"95th={np.percentile(scores, 95):.4f}, 99th={np.percentile(scores, 99):.4f}, "
              f"max={scores.max():.4f}, above_thresh={above}, recall={above/count:.4f}")
              
    # ---------------------------------------------------------
    # 3. EVIDENCE DISTRIBUTION BY ANOMALY TYPE
    # ---------------------------------------------------------
    print("\n[3. EVIDENCE DISTRIBUTION BY ANOMALY TYPE]")
    cols = ['temporal_evidence', 'statistical_evidence', 'multivariate_evidence', 'stability_evidence', 'missing_evidence', 'model_evidence', 'contextual_consistency', 'suppression_factor']
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
        means = subset[cols].mean()
        print(f"{t.upper()} Means:")
        for col, val in means.items():
            print(f"  {col}: {val:.4f}")
            
    # ---------------------------------------------------------
    # 4. SUPPRESSION IMPACT ANALYSIS
    # ---------------------------------------------------------
    print("\n[4. SUPPRESSION IMPACT ANALYSIS]")
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        base = subset['base_score'].values
        final = subset['hybrid_anomaly_score'].values
        supp = subset['suppression_factor'].values
        
        reduction = 1.0 - (final / np.where(base > 0, base, 1.0))
        # Handle all_missing cases where final > base
        reduction = np.clip(reduction, 0.0, 1.0)
        
        r0_10 = ((reduction >= 0.0) & (reduction < 0.10)).sum()
        r10_25 = ((reduction >= 0.10) & (reduction < 0.25)).sum()
        r25_50 = ((reduction >= 0.25) & (reduction < 0.50)).sum()
        r50_75 = ((reduction >= 0.50) & (reduction < 0.75)).sum()
        r75_100 = ((reduction >= 0.75) & (reduction <= 1.00)).sum()
        
        print(f"{t.upper()}:")
        print(f"  Base={base.mean():.4f}, Final={final.mean():.4f}, Supp={supp.mean():.4f}")
        print(f"  Reduction: 0-10%: {r0_10}, 10-25%: {r10_25}, 25-50%: {r25_50}, 50-75%: {r50_75}, 75-100%: {r75_100}")

    # ---------------------------------------------------------
    # 5. THRESHOLD POSITION ANALYSIS
    # ---------------------------------------------------------
    print("\n[5. THRESHOLD POSITION ANALYSIS]")
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.9601]
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        scores = subset['hybrid_anomaly_score'].values
        pcts = [(scores >= thresh).mean() * 100 for thresh in thresholds]
        print(f"{t.upper()}:")
        for th, pct in zip(thresholds, pcts):
            print(f"  >= {th}: {pct:.2f}%")
            
    # ---------------------------------------------------------
    # 6. MISSING-DATA SEPARATION
    # ---------------------------------------------------------
    print("\n[6. MISSING VS NON-MISSING PERFORMANCE]")
    
    # All
    y_true_all = df_hybrid['is_anomaly']
    y_pred_all = df_hybrid['hybrid_prediction']
    cm_all = confusion_matrix(y_true_all, y_pred_all)
    tn, fp, fn, tp = cm_all.ravel()
    prec_all = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec_all = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1_all = f1_score(y_true_all, y_pred_all, zero_division=0)
    
    print("ALL ANOMALIES:")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Precision: {prec_all:.4f}, Recall: {rec_all:.4f}, F1: {f1_all:.4f}")
    
    # Non-Missing Only
    non_missing_mask = (df_hybrid['anomaly_type'] != 'missing') & (df_hybrid['anomaly_type'] != 'normal')
    normal_mask = df_hybrid['anomaly_type'] == 'normal'
    subset_nm = df_hybrid[non_missing_mask | normal_mask]
    
    y_true_nm = subset_nm['is_anomaly']
    y_pred_nm = subset_nm['hybrid_prediction']
    
    cm_nm = confusion_matrix(y_true_nm, y_pred_nm)
    tn, fp, fn, tp = cm_nm.ravel()
    prec_nm = precision_score(y_true_nm, y_pred_nm, zero_division=0)
    rec_nm = recall_score(y_true_nm, y_pred_nm, zero_division=0)
    f1_nm = f1_score(y_true_nm, y_pred_nm, zero_division=0)
    
    print("NON-MISSING ANOMALIES ONLY:")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Precision: {prec_nm:.4f}, Recall: {rec_nm:.4f}, F1: {f1_nm:.4f}")

    # ---------------------------------------------------------
    # 7. MODEL EVIDENCE DOMINATION (PCA CHECK)
    # ---------------------------------------------------------
    print("\n[7. MODEL EVIDENCE DOMINATION CHECK]")
    from ml.anomaly_engine.evidence import ModelEvidenceNormalizer
    
    # We need to fit normalizer on dev again to check exact scales
    df_dev_feat = pd.read_csv('data/processed/aws_dev_features.csv')
    
    from ml.models.statistical_detector import StatisticalBaseline
    from ml.models.pca_detector import PCADetector
    from ml.models.isolation_forest_detector import IsolationForestDetector
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    stat = StatisticalBaseline()
    pca = PCADetector(variance_retained=0.95)
    i_f = IsolationForestDetector(n_estimators=200, random_state=42)
    stat.fit(df_dev_feat, 99.5)
    pca.fit(df_dev_feat, 99.5)
    i_f.fit(df_dev_feat, 99.5)
    s_sc, _ = stat.predict(df_dev_feat)
    p_sc, _ = pca.predict(df_dev_feat)
    i_sc, _ = i_f.predict(df_dev_feat)
    sys.stdout = old_stdout
    
    df_dev_pred = pd.DataFrame({'statistical_score': s_sc, 'pca_score': p_sc, 'isolation_forest_score': i_sc})
    norm = ModelEvidenceNormalizer()
    norm.fit(df_dev_pred)
    
    # Evaluate raw scaled stats on full set
    scores = df_model[['pca_score', 'isolation_forest_score', 'statistical_score']].fillna(0.0)
    scaled = norm.scaler.transform(scores) # order: pca, iso, stat
    
    print("PCA Scaled mean:", scaled[:, 0].mean())
    print("IF Scaled mean:", scaled[:, 1].mean())
    print("STAT Scaled mean:", scaled[:, 2].mean())
    print("PCA Scaled 99th pctl:", np.percentile(scaled[:, 0], 99))
    print("Combined Model Evidence 99th pctl:", np.percentile(df_hybrid['model_evidence'].values, 99))

    # ---------------------------------------------------------
    # 8. EXACT SUPPRESSION FORMULA
    # ---------------------------------------------------------
    print("\n[8. EXACT SUPPRESSION FORMULA]")
    print(f"COHERENT_MULTIVARIATE_THRESHOLD = {COHERENT_MULTIVARIATE_THRESHOLD}")
    print(f"SUPPRESSION_FACTOR = {SUPPRESSION_FACTOR}")
    print("contextual_consistency = (1.0 - m_ev)")
    print("raw_disagreement = df_features['multivariate_z_disagreement'].abs().values")
    print("coherent_mask = (raw_disagreement < COHERENT_MULTIVARIATE_THRESHOLD) & (mis_ev == 0)")
    print("suppression_factor = np.ones(len(df_features))")
    print("suppression_factor[coherent_mask] = SUPPRESSION_FACTOR")
    print("hybrid_score = base_score * suppression_factor")
    print("hybrid_score[all_missing == 1] = np.clip(hybrid_score[all_missing == 1] + 0.8, 0.0, 1.0)")
    print("hybrid_score = np.clip(hybrid_score, 0.0, 1.0)")

    # ---------------------------------------------------------
    # 9. FUSION CONTRIBUTIONS
    # ---------------------------------------------------------
    print("\n[9. FUSION CONTRIBUTIONS]")
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        print(f"{t.upper()} Weighted Contributions:")
        print(f"  Temporal: {(subset['temporal_evidence'] * FUSION_WEIGHTS['temporal']).mean():.4f}")
        print(f"  Statistical: {(subset['statistical_evidence'] * FUSION_WEIGHTS['statistical']).mean():.4f}")
        print(f"  Multivariate: {(subset['multivariate_evidence'] * FUSION_WEIGHTS['multivariate']).mean():.4f}")
        print(f"  Stability: {(subset['stability_evidence'] * FUSION_WEIGHTS['stability']).mean():.4f}")
        print(f"  Missing: {(subset['missing_evidence'] * FUSION_WEIGHTS['missing']).mean():.4f}")
        print(f"  Model: {(subset['model_evidence'] * FUSION_WEIGHTS['model']).mean():.4f}")

    # ---------------------------------------------------------
    # 10. CONFIDENCE ANALYSIS
    # ---------------------------------------------------------
    print("\n[10. CONFIDENCE ANALYSIS]")
    for t in types:
        subset = df_hybrid[df_hybrid['anomaly_type'] == t]
        if len(subset) == 0: continue
            
        conf = subset['anomaly_confidence'].values
        print(f"{t.upper()}:")
        print(f"  Confidence -> mean: {conf.mean():.4f}, median: {np.median(conf):.4f}, max: {conf.max():.4f}")

if __name__ == '__main__':
    run_audit()
