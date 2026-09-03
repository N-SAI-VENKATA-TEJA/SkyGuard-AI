import pandas as pd
from sklearn.ensemble import IsolationForest
from .model_utils import BaselinePreprocessor, calculate_percentile_threshold, apply_threshold

class IsolationForestDetector:
    """
    Isolation Forest Anomaly Detector.
    Uses unsupervised score_samples to derive a custom percentile threshold.
    """
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            contamination='auto'
        )
        self.preprocessor = BaselinePreprocessor()
        self.threshold = None
        
    def fit(self, df: pd.DataFrame, percentile: float = 99.5):
        """Fit Isolation Forest and threshold strictly on development data."""
        # Fit scaler/imputer
        self.preprocessor.fit(df)
        X_scaled = self.preprocessor.transform(df)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        
        # Calculate scores to find threshold
        # score_samples returns opposite of anomaly score (lower = more anomalous).
        # We invert it so higher = more anomalous.
        raw_scores = self.model.score_samples(X_scaled)
        scores = -raw_scores
        
        self.threshold = calculate_percentile_threshold(scores, percentile)
        print(f"IsolationForestDetector fitted.")
        print(f"Threshold (p={percentile}): {self.threshold:.4f}")
        
    def predict(self, df: pd.DataFrame):
        """Transform data and return scores & predictions."""
        if self.threshold is None:
            raise ValueError("Model must be fitted first.")
            
        X_scaled = self.preprocessor.transform(df)
        
        # Invert score: higher = more anomalous
        raw_scores = self.model.score_samples(X_scaled)
        scores = -raw_scores
        
        # Apply strict development-derived threshold (ignore native .predict())
        predictions = apply_threshold(scores, self.threshold)
        
        return scores, predictions
