import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def select_features(df: pd.DataFrame) -> list:
    """
    Selects features by excluding labels, identifiers, and timestamp.
    Retains temporal context, missing indicators, etc.
    """
    exclude_cols = [
        'timestamp', 
        'is_anomaly', 
        'anomaly_type', 
        'affected_sensor', 
        'anomaly_id'
    ]
    
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Filter out completely constant features
    constant_features = [c for c in features if df[c].nunique(dropna=False) <= 1]
    features = [c for c in features if c not in constant_features]
    
    return features

class BaselinePreprocessor:
    """
    Handles causal-friendly batch preprocessing (Imputation and Scaling)
    fitted only on the development dataset.
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.features = None
        
    def fit(self, df: pd.DataFrame):
        """Fit preprocessing parameters ONLY on development data."""
        self.features = select_features(df)
        X = df[self.features]
        
        # Fit imputer and scaler
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations."""
        if self.features is None:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        # Ensure we only transform the exact fitted features
        X = df[self.features]
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled

def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Returns binary predictions based on threshold. 1 = anomaly, 0 = normal."""
    return (scores > threshold).astype(int)

def calculate_percentile_threshold(scores: np.ndarray, percentile: float = 99.5) -> float:
    """Calculates threshold from development scores safely."""
    # Drop NaNs just in case, though imputer should handle features, scores might have NaNs in custom logic
    valid_scores = scores[~np.isnan(scores)]
    return np.percentile(valid_scores, percentile)
