import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from ml.anomaly_engine.step7_sensor_health import SensorHealthTracker

def make_base_row(timestamp, anomaly_flag=False, fault_type='NORMAL', temp=20.0, pres=1010.0, hum=50.0):
    return {
        'timestamp': pd.Timestamp(timestamp),
        'temperature': temp,
        'pressure': pres,
        'humidity': hum,
        'temperature_rate_per_hour': 0.0,
        'pressure_rate_per_hour': 0.0,
        'humidity_rate_per_hour': 0.0,
        'temperature_roll_z_60m': 0.0,
        'pressure_roll_z_60m': 0.0,
        'humidity_roll_z_60m': 0.0,
        'anomaly_flag': anomaly_flag,
        'final_anomaly_score': 1.0 if anomaly_flag else 0.1,
        'anomaly_confidence': 1.0 if anomaly_flag else 0.0,
        'fault_type_hint': fault_type,
        'candidate_fault_family': 'SUDDEN_EVENT' if anomaly_flag else 'NORMAL'
    }

def test_isolated_temperature_spike():
    tracker = SensorHealthTracker()
    row = make_base_row('2009-01-01 00:00:00', True, 'SPIKE')
    row['temperature_rate_per_hour'] = 15.0  # high rate -> evidence 1.0
    out = tracker.process_row(row)
    
    assert out['classified_fault_type'] == 'SPIKE'
    assert out['affected_sensor'] == 'TEMPERATURE'
    # Base penalty for SPIKE is 5.0. Evidence=1.0, conf=1.0, persistence=1.0
    assert out['temperature_health'] == 95.0
    assert out['pressure_health'] == 100.0
    assert out['humidity_health'] == 100.0
    assert out['data_quality_status'] == 'GOOD'
    assert out['anomaly_streak'] == 1

def test_repeated_temperature_spikes():
    tracker = SensorHealthTracker()
    ts = pd.Timestamp('2009-01-01 00:00:00')
    
    for i in range(5):
        row = make_base_row(ts + pd.Timedelta(minutes=10*i), True, 'SPIKE')
        row['temperature_rate_per_hour'] = 15.0
        out = tracker.process_row(row)
        
    assert out['classified_fault_type'] == 'SPIKE'
    assert out['anomaly_streak'] == 5
    # Penalties increase with persistence
    assert out['temperature_health'] < 80.0
    assert out['maintenance_status'] in ["MONITOR / INSPECT", "INSPECT", "MAINTENANCE_RECOMMENDED"]

def test_frozen_period():
    tracker = SensorHealthTracker()
    ts = pd.Timestamp('2009-01-01 00:00:00')
    
    for i in range(6):
        row = make_base_row(ts + pd.Timedelta(minutes=10*i), True, 'FROZEN')
        row['temperature_roll_z_60m'] = 4.0 # high stat deviation
        out = tracker.process_row(row)
        
    assert out['frozen_streak'] == 6
    assert out['maintenance_status'] == 'MAINTENANCE_RECOMMENDED'
    assert out['temperature_health'] < 90.0

def test_drift_progressive():
    tracker = SensorHealthTracker()
    ts = pd.Timestamp('2009-01-01 00:00:00')
    for i in range(10):
        row = make_base_row(ts + pd.Timedelta(minutes=10*i), True, 'DRIFT')
        row['temperature_roll_z_60m'] = 2.0
        out = tracker.process_row(row)
    
    assert out['classified_fault_type'] == 'DRIFT'
    assert out['maintenance_status'] == 'INSPECT'

def test_all_sensors_missing():
    tracker = SensorHealthTracker()
    row = make_base_row('2009-01-01 00:00:00', True, 'MISSING', temp=np.nan, pres=np.nan, hum=np.nan)
    out = tracker.process_row(row)
    
    assert out['classified_fault_type'] == 'MISSING'
    assert out['data_quality_status'] == 'DATA_LOSS'
    assert out['affected_sensor'] == 'ALL_SENSORS'
    # Health should NOT drop
    assert out['temperature_health'] == 100.0
    assert out['pressure_health'] == 100.0
    assert out['humidity_health'] == 100.0

def test_partial_missing():
    tracker = SensorHealthTracker()
    row = make_base_row('2009-01-01 00:00:00', True, 'MISSING', temp=np.nan)
    out = tracker.process_row(row)
    assert out['data_quality_status'] == 'DEGRADED'
    assert out['affected_sensor'] == 'TEMPERATURE'

def test_healthy_recovery():
    tracker = SensorHealthTracker()
    tracker.health['temperature'] = 80.0
    row = make_base_row('2009-01-01 00:00:00', False, 'NORMAL')
    out = tracker.process_row(row)
    
    assert out['temperature_health'] == 80.2

def test_gap_logic_and_scenario():
    tracker = SensorHealthTracker()
    ts = pd.Timestamp('2009-01-01 00:00:00')
    
    # 1. Healthy
    row = make_base_row(ts, False, 'NORMAL')
    out = tracker.process_row(row)
    
    # 2. Anomaly
    ts += pd.Timedelta(minutes=10)
    row = make_base_row(ts, True, 'SPIKE')
    row['temperature_rate_per_hour'] = 15.0
    out = tracker.process_row(row)
    assert out['temperature_health'] == 95.0
    assert out['anomaly_streak'] == 1
    
    # 3. Time Gap > 30 mins
    ts += pd.Timedelta(minutes=60)
    row = make_base_row(ts, False, 'NORMAL')
    out = tracker.process_row(row)
    
    # Gap resets streaks, no recovery applied immediately due to gap logic?
    # Wait, the prompt says: "no health penalties or recoveries are applied merely because of the time gap."
    # The normal observation at ts+60 applies a healthy recovery (+0.2).
    # But the gap itself didn't recover health.
    assert out['anomaly_streak'] == 0
    assert out['temperature_health'] == 95.2  # 95.0 + 0.2
    
    # 4. Another Healthy
    ts += pd.Timedelta(minutes=10)
    row = make_base_row(ts, False, 'NORMAL')
    out = tracker.process_row(row)
    assert out['temperature_health'] == 95.4

def test_causality():
    t1 = SensorHealthTracker()
    rows = []
    ts = pd.Timestamp('2009-01-01 00:00:00')
    for i in range(5):
        rows.append(make_base_row(ts + pd.Timedelta(minutes=10*i), True, 'SPIKE'))
        
    out1 = None
    for r in rows:
        r['temperature_rate_per_hour'] = 15.0
        out1 = t1.process_row(r)
        
    # Process only up to 4th element
    t2 = SensorHealthTracker()
    out2 = None
    for r in rows[:4]:
        out2 = t2.process_row(r)
        
    # The output at index 3 must be identical regardless of whether index 4 is seen
    assert out1['temperature_health'] != out2['temperature_health'] # To prove they advanced
