import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .model_utils import BaselinePreprocessor, calculate_percentile_threshold, apply_threshold

class PCADetector:
    """
    PCA-based Multivariate Anomaly Detector.
    Scores anomalies based on reconstruction error.
    """
    def __init__(self, variance_retained: float = 0.95):
        self.variance_retained = variance_retained
        self.pca = PCA(n_components=self.variance_retained, random_state=42)
        self.preprocessor = BaselinePreprocessor()
        self.threshold = None
        
    def fit(self, df: pd.DataFrame, percentile: float = 99.5):
        """Fit preprocessing, PCA, and threshold only on development data."""
        # Fit scaler/imputer
        self.preprocessor.fit(df)
        X_scaled = self.preprocessor.transform(df)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Calculate scores to find threshold
        scores = self._calculate_reconstruction_error(X_scaled)
        self.threshold = calculate_percentile_threshold(scores, percentile)
        
        print(f"PCADetector fitted. Selected {self.pca.n_components_} components.")
        print(f"Threshold (p={percentile}): {self.threshold:.4f}")
        
    def _calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calculate MSE between original scaled data and PCA reconstruction."""
        X_proj = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_proj)
        
        # Reconstruction error (Higher = more anomalous)
        mse = np.mean(np.square(X - X_reconstructed), axis=1)
        return mse
        
    def predict(self, df: pd.DataFrame):
        """Transform data and return scores & predictions."""
        if self.threshold is None:
            raise ValueError("Model must be fitted first.")
            
        X_scaled = self.preprocessor.transform(df)
        scores = self._calculate_reconstruction_error(X_scaled)
        predictions = apply_threshold(scores, self.threshold)
        
        return scores, predictions
