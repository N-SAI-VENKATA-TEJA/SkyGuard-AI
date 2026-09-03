import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "max_planck_weather_ts.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Core Data Columns mapping
COL_TIMESTAMP = "Date Time"
COL_TEMPERATURE = "T (degC)"
COL_PRESSURE = "p (mbar)"
COL_HUMIDITY = "rh (%)"
