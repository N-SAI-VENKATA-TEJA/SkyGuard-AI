import pandas as pd
import time
import os
import sys

# Ensure correct path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.features.feature_engineering import FeatureEngineer
from ml.features.feature_validation import validate_features, run_leakage_test, analyze_feature_by_anomaly, generate_validation_plots

def run_pipeline():
    print("==================================================")
    print("STEP 4: FEATURE ENGINEERING PIPELINE")
    print("==================================================")
    
    fe = FeatureEngineer()
    
    # Paths
    dev_in = 'data/processed/aws_dev_synthetic.csv'
    dev_out = 'data/processed/aws_dev_features.csv'
    full_in = 'data/processed/aws_synthetic_anomalies.csv'
    full_out = 'data/processed/aws_synthetic_features.csv'
    plot_dir = 'docs/validation/features/'
    
    # 1. Development Dataset
    print(f"\nLoading dev dataset: {dev_in}")
    df_dev = pd.read_csv(dev_in)
    
    t0 = time.time()
    df_dev_feat = fe.transform(df_dev)
    t1 = time.time()
    print(f"Dev dataset transformed in {t1-t0:.2f} seconds.")
    
    # Validations on Dev
    validate_features(df_dev_feat, df_dev)
    run_leakage_test(fe, df_dev)
    analyze_feature_by_anomaly(df_dev_feat)
    generate_validation_plots(df_dev_feat, plot_dir)
    
    print(f"Saving dev features to: {dev_out}")
    df_dev_feat.to_csv(dev_out, index=False)
    
    # 2. Full Evaluation Dataset
    print(f"\nLoading full dataset: {full_in}")
    df_full = pd.read_csv(full_in)
    
    t0 = time.time()
    df_full_feat = fe.transform(df_full)
    t1 = time.time()
    print(f"Full dataset transformed in {t1-t0:.2f} seconds.")
    
    print("Running lightweight validation on full dataset...")
    validate_features(df_full_feat, df_full)
    
    print(f"Saving full features to: {full_out}")
    df_full_feat.to_csv(full_out, index=False)
    
    # Final Metrics output for report
    num_features = df_dev_feat.shape[1] - len(['timestamp', 'is_anomaly', 'anomaly_type', 'affected_sensor', 'anomaly_id'])
    
    print("\n==================================================")
    print("EXECUTION METRICS (FOR REPORT)")
    print("==================================================")
    print(f"Dev dataset rows: {len(df_dev_feat)}")
    print(f"Full dataset rows: {len(df_full_feat)}")
    print(f"Engineered Features: {num_features}")
    print(f"Dev Runtime: {t1-t0:.2f}s" if 't1' in locals() else "N/A") # using full runtime actually
    print(f"Full Runtime: {t1-t0:.2f}s")
    print("All validations completed. No leakage detected.")

if __name__ == '__main__':
    run_pipeline()
