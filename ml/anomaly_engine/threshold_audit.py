import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.anomaly_engine.hybrid_detector_v2 import HybridAnomalyEngineV2
from ml.anomaly_engine.evidence_v2 import *
from ml.anomaly_engine.config_v2 import *
from ml.models.statistical_detector import StatisticalBaseline
from ml.models.pca_detector import PCADetector
from ml.models.isolation_forest_detector import IsolationForestDetector

def run_threshold_audit():
    print("==================================================")
    print("STEP 6 V2 THRESHOLD & SATURATION AUDIT")
    print("==================================================")
    
    # 1. Load DEV data
    df_dev_feat = pd.read_csv('data/processed/aws_dev_features.csv')
    df_dev_labels = pd.read_csv('data/processed/aws_dev_synthetic.csv')
    
    # We need Dev model scores
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    stat_model.fit(df_dev_feat, 99.5)
    pca_model.fit(df_dev_feat, 99.5)
    if_model.fit(df_dev_feat, 99.5)
    
    s_score_dev, _ = stat_model.predict(df_dev_feat)
    p_score_dev, _ = pca_model.predict(df_dev_feat)
    i_score_dev, _ = if_model.predict(df_dev_feat)
    sys.stdout = old_stdout
    
    df_dev_model_scores = pd.DataFrame({
        'statistical_score': s_score_dev,
        'pca_score': p_score_dev,
        'isolation_forest_score': i_score_dev
    })
    
    engine = HybridAnomalyEngineV2()
    engine.model_normalizer.fit(df_dev_model_scores)
    
    n = len(df_dev_feat)
    
    # Extract evidence (similar to hybrid_detector_v2.py)
    tem_ev = extract_temporal_evidence(df_dev_feat)
    sta_ev = extract_statistical_evidence(df_dev_feat)
    mul_ev = extract_multivariate_evidence(df_dev_feat)
    stb_ev = extract_stability_evidence(df_dev_feat)
    drf_ev = extract_drift_evidence(df_dev_feat)
    mod_ev = engine.model_normalizer.transform(df_dev_model_scores)
    
    capped_mod_ev = np.minimum(mod_ev, MODEL_SUPPORT_CAP)
    
    sudden_core = np.maximum.reduce([tem_ev, sta_ev, mul_ev])
    sudden_active = (tem_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                    (sta_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                    (mul_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int)
    sudden_bonus = sudden_active * AGREEMENT_BONUS_WEIGHT
    
    # PRE-CLIP SCORE
    preclip_sudden = sudden_core + sudden_bonus + capped_mod_ev
    postclip_sudden = np.clip(preclip_sudden, 0.0, 1.0)
    
    persistent_core = np.maximum(stb_ev, drf_ev)
    persistent_active = (stb_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                        (drf_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int)
    persistent_bonus = persistent_active * AGREEMENT_BONUS_WEIGHT
    
    preclip_persistent = persistent_core + persistent_bonus + capped_mod_ev
    postclip_persistent = np.clip(preclip_persistent, 0.0, 1.0)
    
    # We apply context factor on postclip to see final family score
    all_mis, any_mis = extract_missing_evidence(df_dev_feat)
    raw_disagreement = df_dev_feat['multivariate_z_disagreement'].abs().fillna(0).values if 'multivariate_z_disagreement' in df_dev_feat.columns else np.zeros(n)
    coherent_mask = (raw_disagreement < COHERENT_MULTIVARIATE_THRESHOLD) & (any_mis == 0)
    context_factor = np.ones(n)
    context_factor[coherent_mask] = CONTEXT_SUPPRESSION_FACTOR
    
    final_sudden = postclip_sudden * context_factor
    final_persistent = postclip_persistent
    
    df_dev_feat['is_anomaly'] = df_dev_labels['is_anomaly']
    df_dev_feat['preclip_sudden'] = preclip_sudden
    df_dev_feat['preclip_persistent'] = preclip_persistent
    df_dev_feat['final_sudden'] = final_sudden
    df_dev_feat['final_persistent'] = final_persistent
    
    def print_stats(series, name):
        s = series.dropna()
        if len(s) == 0:
            print(f"{name}: No data")
            return
        
        frac_1 = (s == 1.0).mean()
        print(f"{name} | Min: {s.min():.4f}, Med: {np.median(s):.4f}, P90: {np.percentile(s, 90):.4f}, P95: {np.percentile(s, 95):.4f}, P99: {np.percentile(s, 99):.4f}, P99.5: {np.percentile(s, 99.5):.4f}, Max: {s.max():.4f} | == 1.0: {frac_1*100:.2f}%")
        
    print("\n--- PRE-CLIP SCORES (SUDDEN) ---")
    print_stats(df_dev_feat['preclip_sudden'], "ALL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 0]['preclip_sudden'], "NORMAL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 1]['preclip_sudden'], "ANOMALY DEV")
    
    print("\n--- PRE-CLIP SCORES (PERSISTENT) ---")
    print_stats(df_dev_feat['preclip_persistent'], "ALL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 0]['preclip_persistent'], "NORMAL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 1]['preclip_persistent'], "ANOMALY DEV")
    
    print("\n--- POST-CLIP FINAL SCORES (SUDDEN) ---")
    print_stats(df_dev_feat['final_sudden'], "ALL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 0]['final_sudden'], "NORMAL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 1]['final_sudden'], "ANOMALY DEV")
    
    print("\n--- POST-CLIP FINAL SCORES (PERSISTENT) ---")
    print_stats(df_dev_feat['final_persistent'], "ALL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 0]['final_persistent'], "NORMAL DEV")
    print_stats(df_dev_feat[df_dev_feat['is_anomaly'] == 1]['final_persistent'], "ANOMALY DEV")

    print("\n==================================================")
    print("EVALUATION RE-RUN METRICS SCRIPT")
    print("==================================================")
    
if __name__ == '__main__':
    run_threshold_audit()
