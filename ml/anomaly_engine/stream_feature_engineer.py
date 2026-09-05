import pandas as pd
import numpy as np
import collections
from datetime import timedelta

class StreamFeatureEngineer:
    def __init__(self):
        # We need up to 360 minutes of history
        self.history = collections.deque()
        self.sensors = ['temperature', 'pressure', 'humidity']
        self.windows_min = [30, 60, 180, 360]
        
        # State for stability features
        self.consec_unchanged = {s: 0.0 for s in self.sensors}
        self.last_obs = None
        
    def _prune_history(self, current_timestamp: pd.Timestamp):
        """Remove observations older than 360 minutes."""
        cutoff = current_timestamp - pd.Timedelta(minutes=360)
        while self.history and self.history[0]['timestamp'] < cutoff:
            self.history.popleft()
            
    def _get_window_history(self, window_min: int, current_timestamp: pd.Timestamp):
        """Get list of observations within the window [current - window_min, current), strictly prior to current."""
        cutoff = current_timestamp - pd.Timedelta(minutes=window_min)
        # Because we prune up to 360m, we just filter the deque
        # Using >= cutoff to perfectly match pandas rolling(..., closed='left')
        return [obs for obs in self.history if obs['timestamp'] >= cutoff]

    def _safe_std(self, vals, ddof=1):
        if len(vals) <= 1:
            return np.nan
        s = np.std(vals, ddof=ddof)
        return 1e-6 if s < 1e-6 else s
        
    def _safe_var(self, vals, ddof=1):
        if len(vals) <= 1:
            return np.nan
        return np.var(vals, ddof=ddof)

    def process_observation(self, obs_dict: dict) -> dict:
        """
        Takes raw observation (timestamp, temperature, pressure, humidity).
        Returns full feature dict matching Step 4.
        """
        ts = pd.to_datetime(obs_dict['timestamp'])
        temp = obs_dict.get('temperature', np.nan)
        pres = obs_dict.get('pressure', np.nan)
        hum = obs_dict.get('humidity', np.nan)
        
        f = {'timestamp': ts, 'temperature': temp, 'pressure': pres, 'humidity': hum}
        
        # 1. Missing Features
        for s in self.sensors:
            f[f'{s}_missing'] = 1 if pd.isna(obs_dict.get(s, np.nan)) else 0
            
        f['any_sensor_missing'] = max(f[f'{s}_missing'] for s in self.sensors)
        f['all_sensors_missing'] = min(f[f'{s}_missing'] for s in self.sensors)
        
        # 2. Time Features
        f['hour'] = ts.hour
        f['minute'] = ts.minute
        f['day_of_week'] = ts.dayofweek
        f['day_of_year'] = ts.dayofyear
        f['month'] = ts.month
        f['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
        
        f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24.0)
        f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24.0)
        f['month_sin'] = np.sin(2 * np.pi * (f['month'] - 1) / 12.0)
        f['month_cos'] = np.cos(2 * np.pi * (f['month'] - 1) / 12.0)
        
        if self.last_obs is not None:
            gap_sec = (ts - self.last_obs['timestamp']).total_seconds()
        else:
            gap_sec = 600.0
            
        f['time_gap_seconds'] = gap_sec
        f['sampling_interval_mins'] = gap_sec / 60.0
        f['time_gap_flag'] = 1 if f['sampling_interval_mins'] > 15.0 else 0
        
        safe_interval = max(f['sampling_interval_mins'], 1.0)
        
        # 3. First-Order Temporal & Stability
        # NOTE: df.diff() computes current - previous
        for s in self.sensors:
            if self.last_obs is not None:
                delta = f[s] - self.last_obs[s]
            else:
                delta = np.nan
                
            f[f'{s}_delta_1'] = delta
            f[f'{s}_abs_delta_1'] = abs(delta) if not pd.isna(delta) else np.nan
            
            if not pd.isna(delta):
                f[f'{s}_rate_per_hour'] = (delta / safe_interval) * 60.0
            else:
                f[f'{s}_rate_per_hour'] = np.nan
                
            # Stability streak (consec_unchanged)
            # Logic: near_zero = (abs_delta < 1e-4)
            # If near_zero, streak increments. If not, streak resets to 0.
            if not pd.isna(delta):
                near_zero = 1 if abs(delta) < 1e-4 else 0
                if near_zero:
                    self.consec_unchanged[s] += 1
                else:
                    self.consec_unchanged[s] = 0
            else:
                self.consec_unchanged[s] = 0
                
            f[f'{s}_consec_unchanged'] = self.consec_unchanged[s]
            
        # Prune history before querying rolling
        self._prune_history(ts)
        
        # 4. Rolling Stats (closed='left' means we only use history, NOT current obs)
        for w_min in self.windows_min:
            win_obs = self._get_window_history(w_min, ts)
            for s in self.sensors:
                vals = [obs[s] for obs in win_obs if not pd.isna(obs[s])]
                
                if len(vals) > 0:
                    r_mean = np.mean(vals)
                    r_std = self._safe_std(vals)
                    r_min = np.min(vals)
                    r_max = np.max(vals)
                    r_med = np.median(vals)
                else:
                    r_mean = r_std = r_min = r_max = r_med = np.nan
                    
                f[f'{s}_roll_mean_{w_min}m'] = r_mean
                f[f'{s}_roll_std_{w_min}m'] = r_std
                f[f'{s}_roll_min_{w_min}m'] = r_min
                f[f'{s}_roll_max_{w_min}m'] = r_max
                f[f'{s}_roll_median_{w_min}m'] = r_med
                
                # Deviations
                dev_mean = f[s] - r_mean
                dev_med = f[s] - r_med
                
                f[f'{s}_dev_mean_{w_min}m'] = dev_mean
                f[f'{s}_dev_med_{w_min}m'] = dev_med
                f[f'{s}_roll_z_{w_min}m'] = dev_mean / r_std if not pd.isna(r_std) else np.nan

        # 5. Stability variance/range (60m)
        win_obs_60 = self._get_window_history(60, ts)
        for s in self.sensors:
            vals = [obs[s] for obs in win_obs_60 if not pd.isna(obs[s])]
            if len(vals) > 0:
                f[f'{s}_roll_var_60m'] = self._safe_var(vals)
                f[f'{s}_roll_range_60m'] = np.max(vals) - np.min(vals)
            else:
                f[f'{s}_roll_var_60m'] = np.nan
                f[f'{s}_roll_range_60m'] = np.nan
                
        # 6. Multivariate Consistency (60m z-scores)
        z_t = f['temperature_roll_z_60m']
        z_p = f['pressure_roll_z_60m']
        z_h = f['humidity_roll_z_60m']
        
        # Max difference in z-scores
        disagreements = []
        if not pd.isna(z_t) and not pd.isna(z_p): disagreements.append(abs(z_t - z_p))
        if not pd.isna(z_t) and not pd.isna(z_h): disagreements.append(abs(z_t - z_h))
        if not pd.isna(z_p) and not pd.isna(z_h): disagreements.append(abs(z_p - z_h))
        f['multivariate_z_disagreement'] = max(disagreements) if disagreements else np.nan
        
        # Number of sensors exhibiting large deviation (> 2 std)
        large_devs = 0
        if not pd.isna(z_t) and abs(z_t) > 2: large_devs += 1
        if not pd.isna(z_p) and abs(z_p) > 2: large_devs += 1
        if not pd.isna(z_h) and abs(z_h) > 2: large_devs += 1
        f['num_sensors_large_dev'] = large_devs
        
        # Dominant sensor z
        z_abs = {'t': abs(z_t) if not pd.isna(z_t) else -1, 
                 'p': abs(z_p) if not pd.isna(z_p) else -1, 
                 'h': abs(z_h) if not pd.isna(z_h) else -1}
        
        if max(z_abs.values()) >= 0:
            winner = max(z_abs, key=z_abs.get)
            f['dominant_sensor_z'] = {'t':0, 'p':1, 'h':2}[winner]
        else:
            f['dominant_sensor_z'] = -1.0 # fallback like fillna(-1)
            
        # 7. Cross-Sensor Temporal Rates
        dt = abs(f['temperature_rate_per_hour']) if not pd.isna(f['temperature_rate_per_hour']) else 0
        dp = abs(f['pressure_rate_per_hour']) if not pd.isna(f['pressure_rate_per_hour']) else 0
        dh = abs(f['humidity_rate_per_hour']) if not pd.isna(f['humidity_rate_per_hour']) else 0
        sum_rates = dt + dp + dh + 1e-6
        f['temp_rate_ratio'] = dt / sum_rates
        f['press_rate_ratio'] = dp / sum_rates
        f['humid_rate_ratio'] = dh / sum_rates
        
        # 8. Robust Features (IQR based)
        for s in self.sensors:
            vals = [obs[s] for obs in win_obs_60 if not pd.isna(obs[s])]
            if len(vals) > 0:
                q75 = np.percentile(vals, 75)
                q25 = np.percentile(vals, 25)
                iqr = q75 - q25
                safe_iqr = 1e-6 if iqr < 1e-6 else iqr
            else:
                iqr = np.nan
                safe_iqr = np.nan
                
            f[f'{s}_roll_iqr_60m'] = iqr
            if not pd.isna(f[f'{s}_dev_med_60m']) and not pd.isna(safe_iqr):
                f[f'{s}_robust_z_60m'] = f[f'{s}_dev_med_60m'] / safe_iqr
            else:
                f[f'{s}_robust_z_60m'] = np.nan
                
        # Update State
        self.last_obs = f
        self.history.append(f)
        
        return f
