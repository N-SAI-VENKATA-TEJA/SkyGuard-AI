import pandas as pd
import numpy as np
import os
import joblib

def validate_artifacts():
    print("==================================================")
    print("STEP 8.1: VALIDATE LOADED ML ARTIFACTS")
    print("==================================================")

    # 1. Load sample dataset
    sample_size = 1000
    print(f"Loading {sample_size} rows from full evaluation dataset...")
    df_features = pd.read_csv('data/processed/aws_synthetic_features.csv', nrows=sample_size)
    
    # Load original batch predictions for comparison
    df_batch = pd.read_csv('data/processed/hybrid_predictions_v2.csv', nrows=sample_size)
    
    # 2. Load artifacts
    print("Loading serialized artifacts from data/artifacts/...")
    stat_model = joblib.load('data/artifacts/statistical_baseline.joblib')
    pca_model = joblib.load('data/artifacts/pca_detector.joblib')
    if_model = joblib.load('data/artifacts/isolation_forest_detector.joblib')
    engine = joblib.load('data/artifacts/hybrid_engine_v2.joblib')
    
    # 3. Generate model scores using loaded models
    print("Generating base model scores using loaded artifacts...")
    s_score, _ = stat_model.predict(df_features)
    p_score, _ = pca_model.predict(df_features)
    i_score, _ = if_model.predict(df_features)
    
    df_model_scores = pd.DataFrame({
        'statistical_score': s_score,
        'pca_score': p_score,
        'isolation_forest_score': i_score
    })
    
    # 4. Generate hybrid predictions using loaded engine
    print("Generating hybrid predictions using loaded V2 engine...")
    df_loaded = engine.predict(df_features, df_model_scores)
    
    # 5. Compare numeric columns (floating point tolerance)
    numeric_cols = [
        'sudden_event_score', 'persistent_fault_score', 'communication_score',
        'final_anomaly_score', 'anomaly_confidence'
    ]
    
    string_cols = [
        'candidate_fault_family', 'fault_type_hint', 'severity', 'primary_evidence'
    ]
    
    mismatch_found = False
    
    print("\nComparing numeric scores (rtol=1e-5)...")
    for col in numeric_cols:
        # np.allclose correctly handles NaNs if equal_nan=True
        # Actually our scores shouldn't have NaNs but just in case
        match = np.allclose(df_batch[col].values, df_loaded[col].values, rtol=1e-5, atol=1e-8, equal_nan=True)
        if not match:
            print(f"FAIL: Mismatch in numeric column '{col}'")
            mismatch_found = True
            
    print("Comparing categorical/string columns...")
    for col in string_cols:
        # Check string equality
        match = (df_batch[col].fillna('') == df_loaded[col].fillna('')).all()
        if not match:
            print(f"FAIL: Mismatch in categorical column '{col}'")
            mismatch_found = True
            
    # Check boolean anomaly flag
    match = (df_batch['anomaly_flag'] == df_loaded['anomaly_flag']).all()
    if not match:
        print("FAIL: Mismatch in 'anomaly_flag'")
        mismatch_found = True

    if not mismatch_found:
        print("\nAll numeric scores, categories, and flags match perfectly.")
        print("VERDICT: PASS")
    else:
        print("\nVERDICT: FAIL")
        
if __name__ == '__main__':
    validate_artifacts()
