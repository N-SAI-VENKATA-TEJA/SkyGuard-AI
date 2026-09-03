import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore') # Suppress sklearn future warnings if any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.statistical_detector import StatisticalBaseline
from ml.models.pca_detector import PCADetector
from ml.models.isolation_forest_detector import IsolationForestDetector
from ml.models.evaluate_models import evaluate_predictions, analyze_per_anomaly_type, generate_validation_plots

def run_pipeline():
    print("==================================================")
    print("STEP 5: UNSUPERVISED ANOMALY DETECTION BASELINE")
    print("==================================================")
    
    dev_in = 'data/processed/aws_dev_features.csv'
    full_in = 'data/processed/aws_synthetic_features.csv'
    out_file = 'data/processed/model_predictions.csv'
    plot_dir = 'docs/validation/models/'
    
    print("\n1. Loading datasets...")
    df_dev = pd.read_csv(dev_in)
    df_full = pd.read_csv(full_in)
    
    print(f"Development Set: {len(df_dev)} rows (used for fitting/thresholds only)")
    print(f"Full Evaluation Set: {len(df_full)} rows (used for inference and metrics)")
    
    # Initialize models
    stat_model = StatisticalBaseline()
    pca_model = PCADetector(variance_retained=0.95)
    if_model = IsolationForestDetector(n_estimators=200, random_state=42)
    
    # Fit models
    print("\n2. Fitting models on Development Dataset...")
    t0_fit = time.time()
    
    stat_model.fit(df_dev, percentile=99.5)
    pca_model.fit(df_dev, percentile=99.5)
    if_model.fit(df_dev, percentile=99.5)
    
    t1_fit = time.time()
    fit_time = t1_fit - t0_fit
    print(f"Total training time: {fit_time:.2f}s")
    
    # Feature count record
    stat_features = len(stat_model.target_features)
    pca_features = len(pca_model.preprocessor.features)
    if_features = len(if_model.preprocessor.features)
    
    # Inference
    print("\n3. Generating predictions on Full Evaluation Dataset...")
    # We must preserve chronological order. The full dataset is already sorted.
    df_preds = df_full[['timestamp', 'is_anomaly', 'anomaly_type', 'affected_sensor', 'anomaly_id', 'temperature', 'pressure', 'humidity']].copy()
    
    t0_inf = time.time()
    
    s_score, s_pred = stat_model.predict(df_full)
    df_preds['statistical_score'] = s_score
    df_preds['statistical_prediction'] = s_pred
    
    p_score, p_pred = pca_model.predict(df_full)
    df_preds['pca_score'] = p_score
    df_preds['pca_prediction'] = p_pred
    
    i_score, i_pred = if_model.predict(df_full)
    df_preds['isolation_forest_score'] = i_score
    df_preds['isolation_forest_prediction'] = i_pred
    
    t1_inf = time.time()
    inf_time = t1_inf - t0_inf
    print(f"Total inference time: {inf_time:.2f}s")
    
    # Save predictions
    df_preds.to_csv(out_file, index=False)
    print(f"\nPredictions saved to {out_file}")
    
    # Evaluation
    print("\n4. Evaluating Models...")
    results_stat = evaluate_predictions(df_preds, 'Statistical', 'statistical')
    results_pca = evaluate_predictions(df_preds, 'PCA', 'pca')
    results_if = evaluate_predictions(df_preds, 'Isolation Forest', 'isolation_forest')
    
    comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'CM'} for r in [results_stat, results_pca, results_if]])
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    print("\nPer-Anomaly Type Analysis (Statistical Baseline):")
    print(analyze_per_anomaly_type(df_preds, 'statistical').to_string(index=False))
    print("\nPer-Anomaly Type Analysis (PCA):")
    print(analyze_per_anomaly_type(df_preds, 'pca').to_string(index=False))
    print("\nPer-Anomaly Type Analysis (Isolation Forest):")
    print(analyze_per_anomaly_type(df_preds, 'isolation_forest').to_string(index=False))
    
    # Generate Plots
    generate_validation_plots(df_preds, {}, plot_dir)
    print(f"\nPlots generated in {plot_dir}")
    
    # False Positive Analysis Output (Quick overview)
    print("\n5. False Positive Overview:")
    fp = df_preds[(df_preds['is_anomaly'] == 0) & 
                  ((df_preds['statistical_prediction'] == 1) | 
                   (df_preds['pca_prediction'] == 1) | 
                   (df_preds['isolation_forest_prediction'] == 1))]
    print(f"Found {len(fp)} normal rows incorrectly flagged by at least one model.")
    
    # Final Metrics output for report parsing
    print("\n==================================================")
    print("EXECUTION METRICS (FOR REPORT)")
    print("==================================================")
    print(f"Statistical Features Used: {stat_features}")
    print(f"PCA Features Used: {pca_features}")
    print(f"IF Features Used: {if_features}")
    print(f"Training Runtime: {fit_time:.2f}s")
    print(f"Inference Runtime: {inf_time:.2f}s")
    
if __name__ == '__main__':
    run_pipeline()
