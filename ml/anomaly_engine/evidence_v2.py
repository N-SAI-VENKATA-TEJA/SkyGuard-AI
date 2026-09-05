import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def robust_bound(x, k=2.0):
    """Bounds positive unbounded values to [0, 1] using exponential decay."""
    return 1.0 - np.exp(-x / k)

class ModelEvidenceNormalizerV2:
    def __init__(self):
        self.scaler = RobustScaler()
        
    def fit(self, df_model_scores: pd.DataFrame):
        # We expect columns: statistical_score, pca_score, isolation_forest_score
        scores = df_model_scores[['pca_score', 'isolation_forest_score', 'statistical_score']].fillna(0.0)
        self.scaler.fit(scores)
        
    def transform(self, df_model_scores: pd.DataFrame) -> np.ndarray:
        scores = df_model_scores[['pca_score', 'isolation_forest_score', 'statistical_score']].fillna(0.0)
        scaled = self.scaler.transform(scores)
        # To avoid the V1 saturation issue, we strictly clip the scaled Z-scores to [0, 10] before bounding
        scaled_clipped = np.clip(scaled, 0.0, 10.0)
        # Average the model evidence
        mean_scaled = scaled_clipped.mean(axis=1)
        # Bound it
        return robust_bound(mean_scaled, k=3.0)

def extract_temporal_evidence(df: pd.DataFrame) -> np.ndarray:
    rates = ['temperature_rate_per_hour', 'pressure_rate_per_hour', 'humidity_rate_per_hour']
    max_rate = df[rates].abs().max(axis=1).fillna(0).values
    # Empirical k=25 based on normal variance
    return robust_bound(max_rate, k=25.0)

def extract_statistical_evidence(df: pd.DataFrame) -> np.ndarray:
    z_cols = ['temperature_roll_z_60m', 'pressure_roll_z_60m', 'humidity_roll_z_60m']
    # Safely handle NaNs and clip extreme Z-scores before bounding
    z_max = df[z_cols].abs().fillna(0.0).clip(upper=50.0).max(axis=1).values
    return robust_bound(z_max, k=30.0)

def extract_multivariate_evidence(df: pd.DataFrame) -> np.ndarray:
    if 'multivariate_z_disagreement' in df.columns:
        disag = df['multivariate_z_disagreement'].abs().fillna(0.0).values
        return robust_bound(disag, k=30.0)
    return np.zeros(len(df))

def extract_stability_evidence(df: pd.DataFrame) -> np.ndarray:
    stab_cols = ['temperature_consec_unchanged', 'pressure_consec_unchanged', 'humidity_consec_unchanged']
    # 6 consecutive unchanged (60 mins) is starting to be highly suspicious
    max_stab = df[stab_cols].fillna(0).max(axis=1).values
    return robust_bound(max_stab, k=6.0)

def extract_drift_evidence(df: pd.DataFrame) -> np.ndarray:
    """
    Detect gradual calibration drift by comparing short-term (60m) rolling mean
    against longer-term (360m) rolling mean. A divergence indicates the sensor
    has drifted from its historical baseline.
    """
    n = len(df)
    drift_signals = []

    for s in ['temperature', 'pressure', 'humidity']:
        # Primary: short-term vs long-term rolling mean divergence
        col_60m = f'{s}_roll_mean_60m'
        col_360m = f'{s}_roll_mean_360m'
        if col_60m in df.columns and col_360m in df.columns:
            divergence = (df[col_60m] - df[col_360m]).abs().fillna(0).values
            drift_signals.append(divergence)

        # Secondary: fallback to legacy diff columns if they exist
        for suffix in ['_diff_roll_mean_24h', '_diff_roll_mean_6h']:
            col = f'{s}{suffix}'
            if col in df.columns:
                drift_signals.append(df[col].abs().fillna(0).values)
                break

    if not drift_signals:
        return np.zeros(n)

    # Combine: take the max signal across all sensors and signal types
    stacked = np.column_stack(drift_signals)
    max_drift = np.nanmax(stacked, axis=1)
    # k=3.5: compromise between old 5.0 (too lenient) and 2.0 (too aggressive)
    return robust_bound(max_drift, k=3.5)

def extract_missing_evidence(df: pd.DataFrame):
    # Returns binary indicators, not a continuous [0,1] bounded score
    all_missing = df['all_sensors_missing'].fillna(0).values
    any_missing = df['any_sensor_missing'].fillna(0).values
    return all_missing, any_missing
