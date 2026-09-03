import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Constructs causal, non-leaking features for AWS data.
    Ensures all rolling features use strictly past data (`shift(1)`).
    """
    
    def __init__(self):
        self.sensors = ['temperature', 'pressure', 'humidity']
        # Windows expressed in minutes
        self.windows_min = [30, 60, 180, 360]
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering pipeline."""
        df_out = df.copy()
        
        # Ensure timestamp is datetime and sorted
        df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
        df_out = df_out.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Missing Features (Group 8)
        # Needs to be done before any interpolation or further math
        df_out = self._add_missing_features(df_out)
        
        # 2. Time & Sampling Features (Group 1)
        df_out = self._add_time_features(df_out)
        
        # 3. First-Order Temporal Features (Group 3)
        df_out = self._add_first_order_features(df_out)
        
        # 4. Causal Rolling Statistics (Group 4)
        df_out = self._add_rolling_features(df_out)
        
        # 5. Stability / Frozen Sensor Features (Group 5)
        df_out = self._add_stability_features(df_out)
        
        # 6. Multivariate Consistency Features (Group 6)
        df_out = self._add_multivariate_features(df_out)
        
        # 7. Cross-Sensor Temporal Features (Group 7)
        df_out = self._add_cross_sensor_features(df_out)
        
        # 8. Sensor-Specific Robust Features (Group 9)
        df_out = self._add_robust_features(df_out)
        
        # Reorder to keep labels at the end
        df_out = self._reorder_columns(df_out)
        return df_out

    def _add_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 8: Explicit indicators for missing data."""
        for sensor in self.sensors:
            df[f'{sensor}_missing'] = df[sensor].isna().astype(int)
        
        df['any_sensor_missing'] = df[[f'{s}_missing' for s in self.sensors]].max(axis=1)
        df['all_sensors_missing'] = df[[f'{s}_missing' for s in self.sensors]].min(axis=1)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 1: Causal time & context features."""
        # Standard time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
        
        # Sampling/Gap features
        df['time_gap_seconds'] = df['timestamp'].diff().dt.total_seconds()
        # For the first row, assume standard 600s gap
        df['time_gap_seconds'] = df['time_gap_seconds'].fillna(600.0) 
        df['sampling_interval_mins'] = df['time_gap_seconds'] / 60.0
        
        # Flag unusually large gaps (> 15 mins)
        df['time_gap_flag'] = (df['sampling_interval_mins'] > 15.0).astype(int)
        
        return df

    def _add_first_order_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 3: Causal temporal changes."""
        for sensor in self.sensors:
            # Change since last observation (causal)
            delta = df[sensor].diff()
            df[f'{sensor}_delta_1'] = delta
            df[f'{sensor}_abs_delta_1'] = delta.abs()
            
            # Rate of change per hour
            # Avoid division by zero, min gap = 1 min
            safe_interval = np.maximum(df['sampling_interval_mins'], 1.0)
            df[f'{sensor}_rate_per_hour'] = (delta / safe_interval) * 60.0
            
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 4: Strictly causal rolling statistics using closed='left'."""
        # Set index for time-based rolling
        df_idx = df.set_index('timestamp')
        
        for w_min in self.windows_min:
            w = f'{w_min}min'
            for sensor in self.sensors:
                # Rolling stats (strictly past only)
                r_causal = df_idx[sensor].rolling(w, closed='left')
                
                mean = r_causal.mean().values
                std = r_causal.std().values
                min_v = r_causal.min().values
                max_v = r_causal.max().values
                med = r_causal.median().values
                
                df[f'{sensor}_roll_mean_{w_min}m'] = mean
                df[f'{sensor}_roll_std_{w_min}m'] = std
                df[f'{sensor}_roll_min_{w_min}m'] = min_v
                df[f'{sensor}_roll_max_{w_min}m'] = max_v
                df[f'{sensor}_roll_median_{w_min}m'] = med
                
                # Derived deviations (Current Observation vs Causal Baseline)
                df[f'{sensor}_dev_mean_{w_min}m'] = df[sensor] - df[f'{sensor}_roll_mean_{w_min}m']
                df[f'{sensor}_dev_med_{w_min}m'] = df[sensor] - df[f'{sensor}_roll_median_{w_min}m']
                
                # Z-Score (handle zero std)
                safe_std = np.where(pd.isna(std) | (std < 1e-6), 1e-6, std)
                df[f'{sensor}_roll_z_{w_min}m'] = df[f'{sensor}_dev_mean_{w_min}m'] / safe_std
                
        return df

    def _add_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 5: Frozen sensor detection features (causal)."""
        for sensor in self.sensors:
            # We want to count how many consecutive times the value has been almost identical
            # Using absolute delta
            delta = df[f'{sensor}_abs_delta_1']
            near_zero = (delta < 1e-4).astype(int)
            
            # Cumulative sum of near_zero streaks. This creates blocks.
            streak = near_zero.groupby((near_zero == 0).cumsum()).cumsum()
            df[f'{sensor}_consec_unchanged'] = streak
            
            # Rolling variance & range (causal via closed='left')
            df_idx = df.set_index('timestamp')
            r_causal = df_idx[sensor].rolling('60min', closed='left')
            df[f'{sensor}_roll_var_60m'] = r_causal.var().values
            df[f'{sensor}_roll_range_60m'] = r_causal.max().values - r_causal.min().values
            
        return df

    def _add_multivariate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 6: Multivariate Consistency."""
        if 'temperature_roll_z_60m' in df.columns:
            z_t = df['temperature_roll_z_60m']
            z_p = df['pressure_roll_z_60m']
            z_h = df['humidity_roll_z_60m']
            
            # Disagreement: max difference in z-scores across sensors
            df['multivariate_z_disagreement'] = np.maximum.reduce([
                np.abs(z_t - z_p),
                np.abs(z_t - z_h),
                np.abs(z_p - z_h)
            ])
            
            # Number of sensors exhibiting large deviation (> 2 std)
            df['num_sensors_large_dev'] = ((np.abs(z_t) > 2).astype(int) + 
                                           (np.abs(z_p) > 2).astype(int) + 
                                           (np.abs(z_h) > 2).astype(int))
            
            # Dominant changing sensor (sensor with highest abs z-score)
            z_abs = pd.DataFrame({'t': np.abs(z_t), 'p': np.abs(z_p), 'h': np.abs(z_h)})
            # 0=t, 1=p, 2=h
            df['dominant_sensor_z'] = z_abs.idxmax(axis=1).map({'t':0, 'p':1, 'h':2}).fillna(-1)
            
        return df

    def _add_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 7: Cross-sensor temporal comparisons."""
        dt = df['temperature_rate_per_hour'].abs()
        dp = df['pressure_rate_per_hour'].abs()
        dh = df['humidity_rate_per_hour'].abs()
        
        sum_rates = dt + dp + dh + 1e-6
        df['temp_rate_ratio'] = dt / sum_rates
        df['press_rate_ratio'] = dp / sum_rates
        df['humid_rate_ratio'] = dh / sum_rates
        
        return df

    def _add_robust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 9: Robust median deviation."""
        df_idx = df.set_index('timestamp')
        for sensor in self.sensors:
            w = '60min'
            r_causal = df_idx[sensor].rolling(w, closed='left')
            # Using IQR instead of MAD for performance
            q75 = r_causal.quantile(0.75).values
            q25 = r_causal.quantile(0.25).values
            iqr = q75 - q25
            df[f'{sensor}_roll_iqr_60m'] = iqr
            
            safe_iqr = np.where(pd.isna(iqr) | (iqr < 1e-6), 1e-6, iqr)
            df[f'{sensor}_robust_z_60m'] = df[f'{sensor}_dev_med_60m'] / safe_iqr
            
        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moves labels to the end."""
        labels = ['is_anomaly', 'anomaly_type', 'affected_sensor', 'anomaly_id']
        existing_labels = [c for c in labels if c in df.columns]
        
        non_labels = [c for c in df.columns if c not in existing_labels]
        return df[non_labels + existing_labels]
