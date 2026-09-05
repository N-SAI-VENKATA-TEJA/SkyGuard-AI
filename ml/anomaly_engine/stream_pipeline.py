import os
import joblib
import pandas as pd
import json

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
        # We need at least 60m of history to compute robust_z_60m correctly
        # But even with less, pandas returns NaN and we compute properly.
        # We define warm-up strictly as: do we have any rolling z-score yet?
        # Actually, let's just see if temperature_roll_z_60m is NaN. 
        # Wait, the prompt says "If Step 6 V2 cannot safely produce a valid prediction... return an explicit warm-up".
        # Step 6 V2 relies heavily on robust_z_60m, roll_z_60m, etc.
        # If they are NaN, the models will impute them (BaselinePreprocessor imputes with median).
        is_warmup = len(self.feature_engineer.history) < 36
        
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
        
        # If in warmup, we should NOT damage health.
        # We can enforce this by forcing anomaly_flag = False during warmup.
        if is_warmup:
            step7_input['anomaly_flag'] = False
            step7_input['fault_type_hint'] = 'NORMAL'
            
        final_out = self.health_tracker.process_row(step7_input)
        
        # 6. Format Result Contract
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
            'explanation': final_out['fault_explanation']
        }
        
        return result
