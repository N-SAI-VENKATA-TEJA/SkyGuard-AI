import pandas as pd
import numpy as np
from .model_utils import calculate_percentile_threshold, apply_threshold

class StatisticalBaseline:
    """
    Statistical Anomaly Detector based on maximum standardized deviation.
    """
    def __init__(self):
        self.threshold = None
        self.target_features = [
            'temperature_roll_z_60m',
            'pressure_roll_z_60m',
            'humidity_roll_z_60m'
        ]
        
    def _calculate_score(self, df: pd.DataFrame) -> np.ndarray:
        """Score is max absolute z-score across sensors."""
        # Ensure features exist
        missing = [f for f in self.target_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for StatisticalBaseline: {missing}")
            
        z_scores = df[self.target_features].abs()
        # Max across the 3 sensors for each row
        # Higher score = more anomalous
        scores = z_scores.max(axis=1).values
        
        # Handle initial NaNs in rolling features
        scores = np.nan_to_num(scores, nan=0.0)
        return scores
        
    def fit(self, df: pd.DataFrame, percentile: float = 99.5):
        """Fit strictly on development data to find threshold."""
        scores = self._calculate_score(df)
        self.threshold = calculate_percentile_threshold(scores, percentile)
        print(f"StatisticalBaseline fitted. Threshold (p={percentile}): {self.threshold:.4f}")
        
    def predict(self, df: pd.DataFrame):
        """Returns continuous scores and binary predictions."""
        if self.threshold is None:
            raise ValueError("Model must be fitted first.")
            
        scores = self._calculate_score(df)
        predictions = apply_threshold(scores, self.threshold)
        
        return scores, predictions
