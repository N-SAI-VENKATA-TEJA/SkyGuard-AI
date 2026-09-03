import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ml import config as ml_config

RANDOM_SEED = 42

ANOMALY_PARAMS = {
    'spike': {
        'temperature': {'min_mag': 3.0, 'max_mag': 6.0},
        'pressure': {'min_mag': 10.0, 'max_mag': 30.0},
        'humidity': {'min_mag': 15.0, 'max_mag': 30.0},
        'min_duration_obs': 1,
        'max_duration_obs': 3
    },
    'drift': {
        'temperature': {'min_total_drift': 4.0, 'max_total_drift': 10.0},
        'pressure': {'min_total_drift': 15.0, 'max_total_drift': 40.0},
        'humidity': {'min_total_drift': 20.0, 'max_total_drift': 40.0},
        'min_duration_obs': 36, # 6 hours
        'max_duration_obs': 144 # 24 hours
    },
    'frozen': {
        'min_duration_obs': 36, # 6 hours
        'max_duration_obs': 144  # 24 hours
    },
    'offset': {
        'temperature': {'min_offset': 3.0, 'max_offset': 8.0},
        'pressure': {'min_offset': 10.0, 'max_offset': 25.0},
        'humidity': {'min_offset': 15.0, 'max_offset': 35.0},
        'min_duration_obs': 18,
        'max_duration_obs': 72
    },
    'noise': {
        'temperature': {'std_dev': 1.5},
        'pressure': {'std_dev': 4.0},
        'humidity': {'std_dev': 10.0},
        'min_duration_obs': 18,
        'max_duration_obs': 72
    },
    'missing': {
        'min_duration_obs': 3,  # 30 mins
        'max_duration_obs': 36  # 6 hours
    },
    'multivariate_inconsistency': {
        'temperature': {'min_offset': 5.0, 'max_offset': 10.0},
        'pressure': {'min_offset': 20.0, 'max_offset': 40.0},
        'humidity': {'min_offset': 30.0, 'max_offset': 50.0},
        'min_duration_obs': 12,
        'max_duration_obs': 48
    }
}
