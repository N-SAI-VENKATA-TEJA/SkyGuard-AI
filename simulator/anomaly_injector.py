import numpy as np
import pandas as pd

class AnomalyInjector:
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        
    def inject_spike(self, series, start_idx, duration, min_mag, max_mag):
        """Injects a sudden short-duration spike."""
        mod_series = series.copy()
        direction = self.rng.choice([1, -1])
        magnitude = self.rng.uniform(min_mag, max_mag)
        
        for i in range(start_idx, min(start_idx + duration, len(series))):
            # For each point in the duration, spike it (usually duration is 1-3)
            mod_series.iloc[i] += direction * magnitude
            
        return mod_series

    def inject_drift(self, series, start_idx, duration, min_total, max_total):
        """Injects a gradual drift over the duration."""
        mod_series = series.copy()
        direction = self.rng.choice([1, -1])
        total_drift = self.rng.uniform(min_total, max_total)
        
        end_idx = min(start_idx + duration, len(series))
        actual_duration = end_idx - start_idx
        if actual_duration <= 0:
            return mod_series
            
        # Linear drift added to each point
        drift_values = np.linspace(0, total_drift, actual_duration) * direction
        mod_series.iloc[start_idx:end_idx] += drift_values
        
        return mod_series

    def inject_frozen(self, series, start_idx, duration):
        """Injects a stuck/frozen sensor value."""
        mod_series = series.copy()
        end_idx = min(start_idx + duration, len(series))
        if start_idx >= len(series):
            return mod_series
            
        # Freeze at the exact value of the start index
        frozen_value = series.iloc[start_idx]
        mod_series.iloc[start_idx:end_idx] = frozen_value
        
        return mod_series

    def inject_offset(self, series, start_idx, duration, min_offset, max_offset):
        """Injects a sudden persistent offset."""
        mod_series = series.copy()
        direction = self.rng.choice([1, -1])
        offset = self.rng.uniform(min_offset, max_offset)
        
        end_idx = min(start_idx + duration, len(series))
        mod_series.iloc[start_idx:end_idx] += (direction * offset)
        
        return mod_series

    def inject_noise(self, series, start_idx, duration, std_dev):
        """Injects abnormal random noise."""
        mod_series = series.copy()
        end_idx = min(start_idx + duration, len(series))
        actual_duration = end_idx - start_idx
        if actual_duration <= 0:
            return mod_series
            
        noise = self.rng.normal(0, std_dev, actual_duration)
        mod_series.iloc[start_idx:end_idx] += noise
        
        return mod_series

    def inject_missing(self, series, start_idx, duration):
        """Injects missing data (NaNs)."""
        mod_series = series.copy()
        end_idx = min(start_idx + duration, len(series))
        
        # We must use float NaN. If integer type, pandas auto-converts, but our cols are float.
        mod_series.iloc[start_idx:end_idx] = np.nan
        
        return mod_series

    def inject_multivariate(self, df, sensor, start_idx, duration, min_offset, max_offset):
        """
        Injects an inconsistency by drastically offsetting one sensor 
        while leaving the others alone.
        """
        mod_df = df.copy()
        direction = self.rng.choice([1, -1])
        offset = self.rng.uniform(min_offset, max_offset)
        
        end_idx = min(start_idx + duration, len(df))
        mod_df.loc[mod_df.index[start_idx:end_idx], sensor] += (direction * offset)
        
        return mod_df
