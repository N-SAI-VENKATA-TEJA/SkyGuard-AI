import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ml.anomaly_engine.config_v2 import *
from ml.anomaly_engine.evidence_v2 import (
    ModelEvidenceNormalizerV2,
    extract_temporal_evidence,
    extract_statistical_evidence,
    extract_multivariate_evidence,
    extract_stability_evidence,
    extract_drift_evidence,
    extract_missing_evidence
)

class HybridAnomalyEngineV2:
    def __init__(self):
        self.model_normalizer = ModelEvidenceNormalizerV2()
        self.sudden_threshold = 0.5
        self.persistent_threshold = 0.5
        
    def fit(self, df_dev_features: pd.DataFrame, df_dev_model_scores: pd.DataFrame):
        """Fit thresholds and normalizers exclusively on development data."""
        # 1. Fit normalizer
        self.model_normalizer.fit(df_dev_model_scores)
        
        # 2. Extract dev scores
        res = self._compute_scores(df_dev_features, df_dev_model_scores)
        
        # 3. Calculate family-specific thresholds
        self.sudden_threshold = np.nanpercentile(res['sudden_event_score'], THRESHOLD_PERCENTILE)
        self.persistent_threshold = np.nanpercentile(res['persistent_fault_score'], THRESHOLD_PERCENTILE)
        
    def _compute_scores(self, df_features: pd.DataFrame, df_model_scores: pd.DataFrame) -> pd.DataFrame:
        """Internal method to compute all family scores and outputs for a dataset."""
        n = len(df_features)
        
        # Extract evidence
        tem_ev = extract_temporal_evidence(df_features)
        sta_ev = extract_statistical_evidence(df_features)
        mul_ev = extract_multivariate_evidence(df_features)
        stb_ev = extract_stability_evidence(df_features)
        drf_ev = extract_drift_evidence(df_features)
        mod_ev = self.model_normalizer.transform(df_model_scores)
        all_mis, any_mis = extract_missing_evidence(df_features)
        
        # Cap model evidence
        capped_mod_ev = np.minimum(mod_ev, MODEL_SUPPORT_CAP)
        
        # ---------------------------------------------------------
        # FAMILY A: SUDDEN EVENT
        # ---------------------------------------------------------
        sudden_core = np.maximum.reduce([tem_ev, sta_ev, mul_ev])
        
        # Count independent strong signals
        sudden_active = (tem_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                        (sta_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                        (mul_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int)
                        
        sudden_bonus = sudden_active * AGREEMENT_BONUS_WEIGHT
        raw_sudden_score = np.clip(sudden_core + sudden_bonus + capped_mod_ev, 0.0, 1.0)
        
        # Contextual Suppression for Sudden Event
        raw_disagreement = df_features['multivariate_z_disagreement'].abs().fillna(0).values if 'multivariate_z_disagreement' in df_features.columns else np.zeros(n)
        
        coherent_mask = (raw_disagreement < COHERENT_MULTIVARIATE_THRESHOLD) & (any_mis == 0)
        context_factor = np.ones(n)
        context_factor[coherent_mask] = CONTEXT_SUPPRESSION_FACTOR
        
        sudden_event_final = raw_sudden_score * context_factor
        
        # ---------------------------------------------------------
        # FAMILY B: PERSISTENT SENSOR FAULT
        # ---------------------------------------------------------
        persistent_core = np.maximum(stb_ev, drf_ev)
        
        persistent_active = (stb_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int) + \
                            (drf_ev >= EVIDENCE_AGREEMENT_THRESHOLD).astype(int)
                            
        persistent_bonus = persistent_active * AGREEMENT_BONUS_WEIGHT
        persistent_fault_score = np.clip(persistent_core + persistent_bonus + capped_mod_ev, 0.0, 1.0)
        
        # NO Contextual suppression for persistent fault!
        
        # ---------------------------------------------------------
        # FAMILY C: DATA LOSS / COMMUNICATION
        # ---------------------------------------------------------
        comm_score = np.zeros(n)
        comm_score[any_mis == 1] = 0.6
        comm_score[all_mis == 1] = 1.0
        
        # ---------------------------------------------------------
        # FINAL SCORE & FAULT CATEGORIZATION
        # ---------------------------------------------------------
        final_anomaly_score = np.maximum.reduce([sudden_event_final, persistent_fault_score, comm_score])
        
        # ---------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------
        # Confidence logic:
        # Strength = final_anomaly_score
        # Agreement = based on winning family
        
        confidence = np.zeros(n)
        
        for i in range(n):
            score = final_anomaly_score[i]
            if score == 0:
                confidence[i] = 1.0 # Confident it is normal
                continue
                
            # Which family won?
            fam_scores = {
                FAMILY_COMM: comm_score[i],
                FAMILY_SUDDEN: sudden_event_final[i],
                FAMILY_PERSISTENT: persistent_fault_score[i]
            }
            winner = max(fam_scores, key=fam_scores.get)
            
            if winner == FAMILY_COMM:
                confidence[i] = 1.0
            elif winner == FAMILY_SUDDEN:
                ag = sudden_active[i]
                ag_val = 1.0 if ag >= 2 else (0.5 if ag == 1 else 0.0)
                confidence[i] = np.clip(0.5 * score + 0.5 * ag_val, 0.0, 1.0)
            else: # PERSISTENT
                ag = persistent_active[i]
                ag_val = 1.0 if ag >= 2 else (0.5 if ag == 1 else 0.0)
                confidence[i] = np.clip(0.5 * score + 0.5 * ag_val, 0.0, 1.0)
        
        # ---------------------------------------------------------
        # ASSEMBLE DATAFRAME
        # ---------------------------------------------------------
        res = pd.DataFrame({
            'timestamp': df_features['timestamp'] if 'timestamp' in df_features.columns else np.arange(n),
            'sudden_event_score': sudden_event_final,
            'persistent_fault_score': persistent_fault_score,
            'communication_score': comm_score,
            'multivariate_score': mul_ev,
            'context_factor': context_factor,
            'final_anomaly_score': final_anomaly_score,
            'anomaly_confidence': confidence,
            'primary_evidence': '',
            'supporting_evidence': ''
        })
        
        return res
        
    def _assign_hints_and_flags(self, df_res: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
        n = len(df_res)
        candidate_family = [""] * n
        fault_hint = [""] * n
        flags = np.zeros(n, dtype=bool)
        severity = ["NORMAL"] * n
        explanations = [""] * n
        
        sudden_scores = df_res['sudden_event_score'].values
        pers_scores = df_res['persistent_fault_score'].values
        comm_scores = df_res['communication_score'].values
        
        tem_ev = extract_temporal_evidence(df_features)
        sta_ev = extract_statistical_evidence(df_features)
        mul_ev = extract_multivariate_evidence(df_features)
        stb_ev = extract_stability_evidence(df_features)
        drf_ev = extract_drift_evidence(df_features)
        
        for i in range(n):
            s_sud = sudden_scores[i]
            s_per = pers_scores[i]
            s_com = comm_scores[i]
            
            flag_sud = s_sud >= self.sudden_threshold
            flag_per = s_per >= self.persistent_threshold
            flag_com = s_com > 0.5
            
            is_anomaly = flag_sud or flag_per or flag_com
            flags[i] = is_anomaly
            
            fam_scores = {
                FAMILY_COMM: s_com,
                FAMILY_SUDDEN: s_sud,
                FAMILY_PERSISTENT: s_per
            }
            winner = max(fam_scores, key=fam_scores.get)
            
            if not is_anomaly:
                candidate_family[i] = FAMILY_NORMAL
                fault_hint[i] = "NORMAL"
                continue
                
            candidate_family[i] = winner
            
            # Fault Type Hints & Explanations
            if winner == FAMILY_COMM:
                fault_hint[i] = HINT_MISSING
                if s_com == 1.0:
                    explanations[i] = "All three sensor values are missing at this timestamp."
                else:
                    explanations[i] = "Partial telemetry loss detected."
                    
            elif winner == FAMILY_PERSISTENT:
                if stb_ev[i] > drf_ev[i]:
                    fault_hint[i] = HINT_FROZEN
                    explanations[i] = "Sensor values remained unchanged for an extended period."
                elif drf_ev[i] > 0.3:
                    fault_hint[i] = HINT_DRIFT
                    explanations[i] = "Sensors exhibit sustained drift from the expected baseline."
                else:
                    fault_hint[i] = HINT_UNKNOWN
                    explanations[i] = "Persistent sensor fault detected."
                    
            elif winner == FAMILY_SUDDEN:
                # Decide based on sub-evidence
                t = tem_ev[i]
                s = sta_ev[i]
                m = mul_ev[i]
                
                if m > t and m > s:
                    fault_hint[i] = HINT_MULTI
                    explanations[i] = "Sensor disagrees strongly with the recent multivariate pattern."
                elif t > s:
                    fault_hint[i] = HINT_SPIKE
                    explanations[i] = "Sensor reading changed abruptly within a short time window."
                else:
                    # High statistical deviation without necessarily high rate (e.g., offset or noise)
                    fault_hint[i] = HINT_OFFSET
                    explanations[i] = "Sensor reading deviation is unusually high relative to its baseline."
                    
                # Contextual explanation appendage
                if df_res['context_factor'].values[i] < 1.0:
                    explanations[i] += " (Anomaly likelihood reduced because sensor movement is consistent with recent multivariate behavior.)"
            
            # Severity mapping
            fs = df_res['final_anomaly_score'].values[i]
            for low, high, sev in SEVERITY_MAPPING:
                if low <= fs < high:
                    severity[i] = sev
                    break
            if fs >= 1.0:
                severity[i] = "CRITICAL"
                
            if winner == FAMILY_COMM and s_com == 1.0:
                severity[i] = "CRITICAL" # Comm loss is operationally critical
                
        df_res['anomaly_flag'] = flags
        df_res['candidate_fault_family'] = candidate_family
        df_res['fault_type_hint'] = fault_hint
        df_res['severity'] = severity
        df_res['primary_evidence'] = explanations
        
        return df_res
        
    def predict(self, df_features: pd.DataFrame, df_model_scores: pd.DataFrame) -> pd.DataFrame:
        res = self._compute_scores(df_features, df_model_scores)
        res = self._assign_hints_and_flags(res, df_features)
        return res
