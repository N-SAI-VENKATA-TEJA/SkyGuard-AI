import pandas as pd
import time
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.anomaly_engine.hybrid_detector_v2 import HybridAnomalyEngineV2

def run_pipeline():
    print("==================================================")
    print("STEP 6 V2: FAULT-FAMILY HYBRID ANOMALY ENGINE")
    print("==================================================")
    
    dev_feat_in = 'data/processed/aws_dev_features.csv'
    full_feat_in = 'data/processed/aws_synthetic_features.csv'
    
    # We will get model scores by fitting Step 5 models on dev_feat_in
    print("\n1. Loading datasets...")
    df_dev_feat = pd.read_csv(dev_feat_in)
    df_full_feat = pd.read_csv(full_feat_in)
    
    # We load evaluation model scores
    df_full_pred = pd.read_csv('data/processed/model_predictions.csv')
    
    print(f"Development Set: {len(df_dev_feat)} rows")
    print(f"Full Evaluation Set: {len(df_full_feat)} rows")
    
    print("\n2. Re-computing Dev Model Scores for strict isolation...")
    from ml.models.statistical_detector import StatisticalBaseline
    from ml.models.pca_detector import PCADetector
    from ml.models.isolation_forest_detector import IsolationForestDetector
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    
    stat_model.fit(df_dev_feat, percentile=99.5)
    pca_model.fit(df_dev_feat, percentile=99.5)
    if_model.fit(df_dev_feat, percentile=99.5)
    
    s_score_dev, _ = stat_model.predict(df_dev_feat)
    p_score_dev, _ = pca_model.predict(df_dev_feat)
    i_score_dev, _ = if_model.predict(df_dev_feat)
    
    sys.stdout = old_stdout
    
    df_dev_model_scores = pd.DataFrame({
        'statistical_score': s_score_dev,
        'pca_score': p_score_dev,
        'isolation_forest_score': i_score_dev
    })
    
    print("\n3. Fitting V2 Engine on Development Dataset...")
    t0 = time.time()
    engine = HybridAnomalyEngineV2()
    engine.fit(df_dev_feat, df_dev_model_scores)
    fit_time = time.time() - t0
    
    print(f"Sudden Threshold (99.5%): {engine.sudden_threshold:.4f}")
    print(f"Persistent Threshold (99.5%): {engine.persistent_threshold:.4f}")
    
    print("\n4. Inference on Full Evaluation Dataset...")
    t0 = time.time()
    # Predict uses df_full_pred directly as it contains pca_score, isolation_forest_score, statistical_score
    df_res = engine.predict(df_full_feat, df_full_pred)
    inf_time = time.time() - t0
    
    # Save predictions
    out_file = 'data/processed/hybrid_predictions_v2.csv'
    df_out = df_full_feat[['timestamp', 'temperature', 'pressure', 'humidity']].copy()
    for col in df_res.columns:
        if col != 'timestamp':
            df_out[col] = df_res[col]
            
    df_out.to_csv(out_file, index=False)
    print(f"Saved V2 predictions to {out_file}")
    
    print(f"\nPerformance: Fit={fit_time:.2f}s, Inference={inf_time:.2f}s, rows/sec={len(df_full_feat)/inf_time if inf_time>0 else 0:.0f}")

if __name__ == '__main__':
    run_pipeline()
