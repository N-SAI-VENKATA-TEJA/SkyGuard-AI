import pandas as pd
import numpy as np

# Thresholds for operational interpretation
STATUS_THRESHOLDS = {
    'HEALTHY': 90.0,
    'WATCH': 70.0,
    'DEGRADED': 40.0,
    'CRITICAL': 0.0
}

# Max persistence multiplier
MAX_PERSISTENCE_FACTOR = 5.0
PERSISTENCE_GROWTH = 0.5  # Growth per consecutive anomalous observation

# Base penalties
BASE_PENALTIES = {
    'SPIKE': 5.0,
    'NOISE': 2.0,
    'OFFSET': 10.0,
    'DRIFT': 2.0,
    'FROZEN': 5.0,
    'MULTIVARIATE_INCONSISTENCY': 3.0,
    'MISSING': 0.0,
    'UNKNOWN': 1.0,
    'NORMAL': 0.0
}

# Base recovery per healthy observation
BASE_RECOVERY = 0.2

class SensorHealthTracker:
    def __init__(self):
        # Health state (0-100)
        self.health = {
            'temperature': 100.0,
            'pressure': 100.0,
            'humidity': 100.0
        }
        
        # Persistence tracking
        self.streaks = {
            'anomaly': 0,
            'temperature_fault': 0,
            'pressure_fault': 0,
            'humidity_fault': 0,
            'missing': 0,
            'frozen': 0
        }
        
        self.last_timestamp = None

    def _determine_status(self, health_score):
        if health_score >= STATUS_THRESHOLDS['HEALTHY']: return 'HEALTHY'
        if health_score >= STATUS_THRESHOLDS['WATCH']: return 'WATCH'
        if health_score >= STATUS_THRESHOLDS['DEGRADED']: return 'DEGRADED'
        return 'CRITICAL'

    def _get_sensor_evidence(self, row, sensor):
        """
        Compute sensor-specific evidence combining temporal and statistical deviations.
        """
        rate = abs(row.get(f'{sensor}_rate_per_hour', 0.0))
        z_score = abs(row.get(f'{sensor}_roll_z_60m', 0.0))
        is_missing = pd.isna(row.get(sensor, np.nan))
        
        if is_missing:
            return 1.0  # Max evidence if sensor is completely missing
            
        # Combine normalized rate and z-score heuristically
        # Rate > 5 or z_score > 3 means strong evidence
        rate_norm = min(1.0, rate / 10.0)
        z_norm = min(1.0, z_score / 4.0)
        
        return min(1.0, max(rate_norm, z_norm))

    def _determine_attribution(self, row, fault_type):
        """
        Determine which sensor is affected based on evidence.
        """
        if fault_type == 'NORMAL':
            return 'NONE'
        if fault_type == 'MISSING':
            if pd.isna(row.get('temperature')) and pd.isna(row.get('pressure')) and pd.isna(row.get('humidity')):
                return 'ALL_SENSORS'
                
        # Collect evidence for each sensor
        ev_temp = self._get_sensor_evidence(row, 'temperature')
        ev_pres = self._get_sensor_evidence(row, 'pressure')
        ev_hum = self._get_sensor_evidence(row, 'humidity')
        
        evidence_scores = {'TEMPERATURE': ev_temp, 'PRESSURE': ev_pres, 'HUMIDITY': ev_hum}
        
        # High evidence threshold
        implicated = {k: v for k, v in evidence_scores.items() if v > 0.5}
        
        if len(implicated) == 0:
            return 'UNKNOWN'
        elif len(implicated) == 1:
            return list(implicated.keys())[0]
        else:
            if fault_type == 'MULTIVARIATE_INCONSISTENCY':
                return 'MULTIPLE'
            
            # Find the strictly dominant sensor
            sorted_ev = sorted(implicated.items(), key=lambda item: item[1], reverse=True)
            if sorted_ev[0][1] > sorted_ev[1][1] * 1.5:  # 50% stronger than runner up
                return sorted_ev[0][0]
            else:
                return 'MULTIPLE'

    def _determine_maintenance_recommendation(self, fault_type, attribution, anomaly_flag):
        if not anomaly_flag:
            return "NO_ACTION"
            
        if fault_type == 'MISSING':
            return "Check AWS communication/telemetry connection."
            
        if fault_type == 'FROZEN' and self.streaks['frozen'] > 3:
            return "MAINTENANCE_RECOMMENDED"
            
        if fault_type == 'DRIFT':
            return "INSPECT"
            
        if self.streaks['anomaly'] > 3:
            return "MONITOR / INSPECT"
            
        return "MONITOR"

    def _generate_explanation(self, fault_type, attribution, confidence, row):
        if fault_type == 'NORMAL':
            return "Normal operation."
            
        if fault_type == 'MISSING':
            return "Data-loss event: observations are unavailable. This indicates a communication/data-quality issue, not confirmed sensor hardware failure."
            
        if attribution in ['TEMPERATURE', 'PRESSURE', 'HUMIDITY']:
            if fault_type == 'SPIKE':
                return f"Probable {attribution.lower()} spike: reading changed abruptly relative to its recent baseline. Confidence: {confidence:.2f}."
            elif fault_type == 'FROZEN':
                return f"Candidate frozen {attribution.lower()} sensor: reading remained unchanged for multiple intervals. Confidence: {confidence:.2f}."
            elif fault_type == 'DRIFT':
                return f"Likely persistent drift in {attribution.lower()}: steady deviation from baseline. Confidence: {confidence:.2f}."
            elif fault_type == 'OFFSET':
                return f"Possible {attribution.lower()} offset: sudden persistent level shift. Confidence: {confidence:.2f}."
            elif fault_type == 'NOISE':
                return f"High noise in {attribution.lower()} sensor. Confidence: {confidence:.2f}."
                
        if fault_type == 'MULTIVARIATE_INCONSISTENCY':
            return f"Review multivariate sensor consistency: contradictory physical relationships detected. Confidence: {confidence:.2f}."
            
        return f"Anomaly detected with insufficient evidence for specific fault isolation. Confidence: {confidence:.2f}."

    def process_row(self, row_dict):
        ts = pd.to_datetime(row_dict['timestamp'])
        
        # 1. Natural Time Gap Handling
        if self.last_timestamp is not None:
            gap_minutes = (ts - self.last_timestamp).total_seconds() / 60.0
            if gap_minutes > 30.0:
                # Reset persistence counters due to missing continuity
                for k in self.streaks:
                    self.streaks[k] = 0
                    
        self.last_timestamp = ts
        
        anomaly_flag = bool(row_dict.get('anomaly_flag', False))
        fault_hint = row_dict.get('fault_type_hint', 'NORMAL')
        confidence = float(row_dict.get('anomaly_confidence', 0.0))
        
        # 2. Fault Classification
        if not anomaly_flag:
            fault_type = 'NORMAL'
        else:
            fault_type = fault_hint
            
        # 3. Data Quality Status
        temp_miss = pd.isna(row_dict.get('temperature'))
        pres_miss = pd.isna(row_dict.get('pressure'))
        hum_miss = pd.isna(row_dict.get('humidity'))
        
        if temp_miss and pres_miss and hum_miss:
            data_quality = 'DATA_LOSS'
            fault_type = 'MISSING'
        elif temp_miss or pres_miss or hum_miss:
            data_quality = 'DEGRADED'
            fault_type = 'MISSING'
        else:
            data_quality = 'GOOD'
            
        # 4. Sensor Attribution
        attribution = self._determine_attribution(row_dict, fault_type)
        
        # 5. Persistence Tracking
        if anomaly_flag:
            self.streaks['anomaly'] += 1
            if fault_type == 'MISSING':
                self.streaks['missing'] += 1
            if fault_type == 'FROZEN':
                self.streaks['frozen'] += 1
                
            if attribution == 'TEMPERATURE':
                self.streaks['temperature_fault'] += 1
            elif attribution == 'PRESSURE':
                self.streaks['pressure_fault'] += 1
            elif attribution == 'HUMIDITY':
                self.streaks['humidity_fault'] += 1
            elif attribution == 'ALL_SENSORS' or attribution == 'MULTIPLE':
                self.streaks['temperature_fault'] += 1
                self.streaks['pressure_fault'] += 1
                self.streaks['humidity_fault'] += 1
        else:
            # Only healthy observation resets anomaly streak
            self.streaks['anomaly'] = 0
            self.streaks['missing'] = 0
            self.streaks['frozen'] = 0
            self.streaks['temperature_fault'] = 0
            self.streaks['pressure_fault'] = 0
            self.streaks['humidity_fault'] = 0

        # 6. Health Updates
        out_health = {}
        out_status = {}
        
        sensors = ['temperature', 'pressure', 'humidity']
        for sensor in sensors:
            streak = self.streaks[f'{sensor}_fault']
            
            # Is this sensor currently implicated?
            is_implicated = False
            if attribution == sensor.upper() or attribution in ['MULTIPLE', 'ALL_SENSORS']:
                is_implicated = True
                
            # If healthy observation, recover
            if not anomaly_flag and streak == 0:
                recovery = BASE_RECOVERY
                penalty = 0.0
            else:
                recovery = 0.0
                if is_implicated:
                    base_penalty = BASE_PENALTIES.get(fault_type, 1.0)
                    evidence_str = self._get_sensor_evidence(row_dict, sensor)
                    
                    # Missing data does not automatically reduce physical hardware health
                    if fault_type == 'MISSING':
                        penalty = 0.0
                    else:
                        persistence_factor = min(MAX_PERSISTENCE_FACTOR, 1.0 + PERSISTENCE_GROWTH * max(0, streak - 1))
                        penalty = base_penalty * evidence_str * confidence * persistence_factor
                else:
                    penalty = 0.0

            # Apply stateful update
            new_health = self.health[sensor] + recovery - penalty
            new_health = max(0.0, min(100.0, new_health))
            self.health[sensor] = new_health
            
            out_health[f'{sensor}_health'] = new_health
            out_status[f'{sensor}_status'] = self._determine_status(new_health)
            
        # 7. Recommendations and Explanations
        recommendation = self._determine_maintenance_recommendation(fault_type, attribution, anomaly_flag)
        explanation = self._generate_explanation(fault_type, attribution, confidence, row_dict)
        
        # Build Output
        out = {
            'timestamp': row_dict['timestamp'],
            'temperature': row_dict.get('temperature'),
            'pressure': row_dict.get('pressure'),
            'humidity': row_dict.get('humidity'),
            'anomaly_flag': anomaly_flag,
            'final_anomaly_score': row_dict.get('final_anomaly_score'),
            'anomaly_confidence': confidence,
            'candidate_fault_family': row_dict.get('candidate_fault_family'),
            'fault_type_hint': fault_hint,
            
            'classified_fault_type': fault_type,
            'affected_sensor': attribution,
            
            'temperature_health': out_health['temperature_health'],
            'pressure_health': out_health['pressure_health'],
            'humidity_health': out_health['humidity_health'],
            
            'temperature_status': out_status['temperature_status'],
            'pressure_status': out_status['pressure_status'],
            'humidity_status': out_status['humidity_status'],
            
            'data_quality_status': data_quality,
            
            'anomaly_streak': self.streaks['anomaly'],
            'missing_streak': self.streaks['missing'],
            'frozen_streak': self.streaks['frozen'],
            'sensor_fault_streak': max(self.streaks['temperature_fault'], self.streaks['pressure_fault'], self.streaks['humidity_fault']),
            
            'maintenance_status': recommendation,
            'fault_explanation': explanation
        }
        
        return out
