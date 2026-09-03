import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def extract_temporal_evidence(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts temporal evidence based on causal rate of change.
    Bounded [0,1] using 1 - exp(-|rate|/k).
    """
    # Use max absolute rate per hour across sensors
    rates = df[['temperature_rate_per_hour', 'pressure_rate_per_hour', 'humidity_rate_per_hour']].abs()
    max_rate = rates.max(axis=1).values
    max_rate = np.nan_to_num(max_rate, nan=0.0)
    
    # K parameter chosen heuristically to map large natural rates (e.g. 5) to ~0.5, and extreme (20+) to ~1.0
    k = 10.0
    evidence = 1.0 - np.exp(-max_rate / k)
    return np.clip(evidence, 0.0, 1.0)

def extract_statistical_evidence(df: pd.DataFrame, max_z: float = 100.0) -> np.ndarray:
    """
    Extracts statistical evidence from 60m rolling Z-scores.
    Caps extreme scores and bounds [0,1].
    """
    z_scores = df[['temperature_roll_z_60m', 'pressure_roll_z_60m', 'humidity_roll_z_60m']].abs()
    max_z_score = z_scores.max(axis=1).values
    max_z_score = np.nan_to_num(max_z_score, nan=0.0)
    
    # Cap to prevent 14M blow-ups
    max_z_score = np.clip(max_z_score, 0.0, max_z)
    
    # Map Z-scores to [0,1]
    # Z=0 -> 0, Z=3 -> ~0.5, Z=10+ -> ~1.0
    k = 4.0
    evidence = 1.0 - np.exp(-max_z_score / k)
    return np.clip(evidence, 0.0, 1.0)

def extract_multivariate_evidence(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts multivariate evidence using disagreement.
    """
    disagreement = df['multivariate_z_disagreement'].abs().values
    disagreement = np.nan_to_num(disagreement, nan=0.0)
    
    # Disagreement 0 -> 0, Disagreement 3 -> ~0.5, Disagreement 10+ -> ~1.0
    k = 4.0
    evidence = 1.0 - np.exp(-disagreement / k)
    return np.clip(evidence, 0.0, 1.0)

def extract_stability_evidence(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts stability (frozen) evidence based on consec_unchanged.
    """
    consec = df[['temperature_consec_unchanged', 'pressure_consec_unchanged', 'humidity_consec_unchanged']]
    max_consec = consec.max(axis=1).values
    max_consec = np.nan_to_num(max_consec, nan=0.0)
    
    # 0 mins -> 0. 60 mins -> ~0.5. 300+ mins -> ~1.0
    k = 80.0
    evidence = 1.0 - np.exp(-max_consec / k)
    return np.clip(evidence, 0.0, 1.0)

def extract_missing_evidence(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts missing/communication evidence.
    """
    all_missing = df['all_sensors_missing'].values
    any_missing = df['any_sensor_missing'].values
    
    evidence = np.zeros(len(df))
    evidence[any_missing == 1] = 0.5
    evidence[all_missing == 1] = 1.0
    return evidence

class ModelEvidenceNormalizer:
    """
    Normalizes Step 5 model scores robustly using ONLY development data.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        
    def fit(self, df_preds: pd.DataFrame):
        # We need the scores of PCA, IF, and Statistical from dev
        # For simplicity, we just take the pre-computed scores and fit the scaler
        # df_preds should contain 'pca_score', 'isolation_forest_score', 'statistical_score'
        scores = df_preds[['pca_score', 'isolation_forest_score', 'statistical_score']]
        scores = scores.fillna(0.0)
        self.scaler.fit(scores)
        
    def transform(self, df_preds: pd.DataFrame) -> np.ndarray:
        scores = df_preds[['pca_score', 'isolation_forest_score', 'statistical_score']]
        scores = scores.fillna(0.0)
        scaled = self.scaler.transform(scores)
        
        # Scale values using a sigmoid function to bound [0,1]
        # Scaled values are roughly N(0,1) around the median. We want high positive values to go to 1.
        # scaled is a 3-column array. We mean them.
        mean_scaled = np.mean(scaled, axis=1)
        evidence = 1.0 / (1.0 + np.exp(-mean_scaled))
        
        # This gives ~0.5 for median normal behavior, which is a bit high. 
        # Better: Shift it so median is 0, and map positive outliers to [0,1].
        # Clip negative to 0.
        mean_scaled = np.clip(mean_scaled, 0.0, None)
        # Apply exponential bounding
        evidence = 1.0 - np.exp(-mean_scaled / 2.0)
        
        return np.clip(evidence, 0.0, 1.0)
