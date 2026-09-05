import os
import joblib
import pandas as pd
import numpy as np
import json
import shap

from ml.anomaly_engine.stream_feature_engineer import StreamFeatureEngineer
from ml.anomaly_engine.step7_sensor_health import SensorHealthTracker

class StreamPipeline:
    def __init__(self, artifact_dir='data/artifacts'):
        self.artifact_dir = artifact_dir
        
        # Load Frozen Artifacts
        self.stat_model = joblib.load(os.path.join(artifact_dir, 'statistical_baseline.joblib'))
        self.pca_model = joblib.load(os.path.join(artifact_dir, 'pca_detector.joblib'))
        self.if_model = joblib.load(os.path.join(artifact_dir, 'isolation_forest_detector.joblib'))
        self.hybrid_engine = joblib.load(os.path.join(artifact_dir, 'hybrid_engine_v2.joblib'))
        
        with open(os.path.join(artifact_dir, 'manifest.json'), 'r') as f:
            self.manifest = json.load(f)
            
        self.feature_cols = self.manifest['feature_column_names']
        
        # Initialize Stateful Components
        self.feature_engineer = StreamFeatureEngineer()
        self.health_tracker = SensorHealthTracker()
        
        # Initialize SHAP Explainer for Isolation Forest
        # TreeExplainer works natively with sklearn IsolationForest
        try:
            self.shap_explainer = shap.TreeExplainer(self.if_model.model)
        except Exception:
            self.shap_explainer = None
        
    def _compute_shap_features(self, df_row: pd.DataFrame) -> list:
        """Compute top-3 SHAP feature contributions for explainability."""
        if self.shap_explainer is None:
            return []
        
        try:
            X_scaled = self.if_model.preprocessor.transform(df_row)
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            # shap_values shape: (1, n_features) — get the single row
            sv = shap_values[0]
            
            # Get feature names (exclude timestamp if present)
            feature_names = [c for c in self.feature_cols if c != 'timestamp']
            
            # Match lengths (preprocessor may have dropped timestamp)
            if len(sv) == len(feature_names):
                indices = np.argsort(np.abs(sv))[::-1][:3]
                return [
                    {"feature": feature_names[i], "contribution": round(float(sv[i]), 4)}
                    for i in indices
                ]
            elif len(sv) == len(self.feature_cols):
                indices = np.argsort(np.abs(sv))[::-1][:3]
                return [
                    {"feature": self.feature_cols[i], "contribution": round(float(sv[i]), 4)}
                    for i in indices
                ]
        except Exception:
            pass
        
        return []
    
    def _compute_suggested_corrections(self) -> dict:
        """Compute corrected values from rolling median of the history buffer."""
        corrections = {}
        history = self.feature_engineer.history
        
        if len(history) < 3:
            return corrections
            
        for sensor in ['temperature', 'pressure', 'humidity']:
            vals = [obs[sensor] for obs in history if not pd.isna(obs.get(sensor, np.nan))]
            if vals:
                corrections[sensor] = round(float(np.median(vals)), 2)
                
        return corrections
        
    def process_observation(self, timestamp, temperature, pressure, humidity) -> dict:
        obs = {
            'timestamp': timestamp,
            'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity
        }
        
        # 1. Stateful Feature Engineering
        features = self.feature_engineer.process_observation(obs)
        
        # Determine Warm-Up
        # Reduced from 36 (6 hours) to 12 (2 hours) for faster operational readiness.
        # Basic fault detection (MISSING, FROZEN) still works during warmup.
        is_warmup = len(self.feature_engineer.history) < 12
        
        # 2. Convert to DataFrame to pass to sklearn models
        # Ensure exact column ordering
        df_row = pd.DataFrame([features], columns=self.feature_cols)
        
        # 3. Model Inference
        # (models safely handle NaNs via their imputers, but we want the actual outputs)
        s_score, _ = self.stat_model.predict(df_row)
        p_score, _ = self.pca_model.predict(df_row)
        i_score, _ = self.if_model.predict(df_row)
        
        df_scores = pd.DataFrame({
            'statistical_score': s_score,
            'pca_score': p_score,
            'isolation_forest_score': i_score
        })
        
        # 4. Hybrid Engine Inference
        df_res = self.hybrid_engine.predict(df_row, df_scores)
        
        # 5. Step 7 Sensor Health Tracker
        # Prepare the input dict for Step 7
        res_dict = df_res.iloc[0].to_dict()
        # Merge with raw features
        step7_input = {**features, **res_dict}
        
        # During warmup: allow MISSING and FROZEN detection, suppress everything else.
        # This lets the system catch obvious faults even before full history is available.
        if is_warmup:
            fault_hint = step7_input.get('fault_type_hint', 'NORMAL')
            if fault_hint not in ('MISSING', 'FROZEN'):
                step7_input['anomaly_flag'] = False
                step7_input['fault_type_hint'] = 'NORMAL'
            
        final_out = self.health_tracker.process_row(step7_input)
        
        # 6. SHAP Explainability (compute only when anomaly is detected)
        shap_top_features = []
        if final_out['anomaly_flag'] and not is_warmup:
            shap_top_features = self._compute_shap_features(df_row)
        
        # 7. Suggested Corrections (compute only when anomaly is detected)
        suggested_corrections = None
        if final_out['anomaly_flag']:
            suggested_corrections = self._compute_suggested_corrections()
        
        # 8. Format Result Contract
        result = {
            'timestamp': str(timestamp),
            'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity,
            
            'processing_state': 'WARMUP' if is_warmup else 'PROCESSED',
            
            'anomaly_score': final_out['final_anomaly_score'],
            'anomaly_flag': final_out['anomaly_flag'],
            'severity': df_res['severity'].values[0] if not is_warmup else 'NORMAL',
            'confidence': final_out['anomaly_confidence'],
            
            'fault_type': final_out['classified_fault_type'],
            'affected_sensor': final_out['affected_sensor'],
            
            'sensor_health_temperature': final_out['temperature_health'],
            'sensor_health_pressure': final_out['pressure_health'],
            'sensor_health_humidity': final_out['humidity_health'],
            
            'temperature_status': final_out['temperature_status'],
            'pressure_status': final_out['pressure_status'],
            'humidity_status': final_out['humidity_status'],
            
            'data_quality_status': final_out['data_quality_status'],
            
            'maintenance_status': final_out['maintenance_status'],
            'explanation': final_out['fault_explanation'],
            
            'shap_top_features': shap_top_features,
            'suggested_corrections': suggested_corrections
        }
        
        return result
