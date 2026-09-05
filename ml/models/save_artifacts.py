import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime, timezone

from ml.models.statistical_detector import StatisticalBaseline
from ml.models.pca_detector import PCADetector
from ml.models.isolation_forest_detector import IsolationForestDetector
from ml.anomaly_engine.hybrid_detector_v2 import HybridAnomalyEngineV2

def save_artifacts():
    print("==================================================")
    print("STEP 8.1: FREEZE & SAVE ML ARTIFACTS")
    print("==================================================")

    out_dir = 'data/artifacts'
    os.makedirs(out_dir, exist_ok=True)

    dev_feat_in = 'data/processed/aws_dev_features.csv'
    print(f"Loading development data from {dev_feat_in}...")
    df_dev_feat = pd.read_csv(dev_feat_in)
    
    # 1. Fit Base Models
    print("\n1. Fitting Step 5 Base Models (Percentile = 99.5)...")
    
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    
    stat_model.fit(df_dev_feat, percentile=99.5)
    pca_model.fit(df_dev_feat, percentile=99.5)
    if_model.fit(df_dev_feat, percentile=99.5)
    
    # 2. Extract Dev Model Scores
    s_score_dev, _ = stat_model.predict(df_dev_feat)
    p_score_dev, _ = pca_model.predict(df_dev_feat)
    i_score_dev, _ = if_model.predict(df_dev_feat)
    
    df_dev_model_scores = pd.DataFrame({
        'statistical_score': s_score_dev,
        'pca_score': p_score_dev,
        'isolation_forest_score': i_score_dev
    })
    
    # 3. Fit V2 Hybrid Engine
    print("\n2. Fitting Step 6 V2 Hybrid Engine...")
    engine = HybridAnomalyEngineV2()
    engine.fit(df_dev_feat, df_dev_model_scores)
    
    # 4. Save Artifacts via joblib
    print("\n3. Serializing artifacts...")
    
    stat_file = os.path.join(out_dir, 'statistical_baseline.joblib')
    pca_file = os.path.join(out_dir, 'pca_detector.joblib')
    if_file = os.path.join(out_dir, 'isolation_forest_detector.joblib')
    engine_file = os.path.join(out_dir, 'hybrid_engine_v2.joblib')
    
    joblib.dump(stat_model, stat_file)
    joblib.dump(pca_model, pca_file)
    joblib.dump(if_model, if_file)
    joblib.dump(engine, engine_file)
    
    # 5. Create Manifest
    manifest = {
        'creation_timestamp': datetime.now(timezone.utc).isoformat(),
        'dataset_used_for_fitting': dev_feat_in,
        'row_count': len(df_dev_feat),
        'feature_count': len(df_dev_feat.columns),
        'feature_column_names': list(df_dev_feat.columns),
        'model_configurations': {
            'statistical_baseline': {
                'percentile': 99.5,
                'threshold': float(stat_model.threshold)
            },
            'pca_detector': {
                'variance_retained': 0.95,
                'percentile': 99.5,
                'threshold': float(pca_model.threshold),
                'random_state': 42
            },
            'isolation_forest_detector': {
                'n_estimators': 200,
                'percentile': 99.5,
                'threshold': float(if_model.threshold),
                'random_state': 42
            },
            'hybrid_engine_v2': {
                'sudden_threshold': float(engine.sudden_threshold),
                'persistent_threshold': float(engine.persistent_threshold)
            }
        },
        'random_seed': 42,
        'artifacts': {
            'statistical_baseline': 'statistical_baseline.joblib',
            'pca_detector': 'pca_detector.joblib',
            'isolation_forest_detector': 'isolation_forest_detector.joblib',
            'hybrid_engine_v2': 'hybrid_engine_v2.joblib'
        }
    }
    
    manifest_file = os.path.join(out_dir, 'manifest.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nSaved all artifacts and manifest to {out_dir}")

if __name__ == '__main__':
    save_artifacts()
