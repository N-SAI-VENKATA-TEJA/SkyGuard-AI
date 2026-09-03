import pandas as pd
import numpy as np

from .config import FUSION_WEIGHTS, MAX_Z_CAP, SUPPRESSION_FACTOR, COHERENT_MULTIVARIATE_THRESHOLD
from .evidence import (
    extract_temporal_evidence,
    extract_statistical_evidence,
    extract_multivariate_evidence,
    extract_stability_evidence,
    extract_missing_evidence,
    ModelEvidenceNormalizer
)

class HybridAnomalyEngine:
    def __init__(self):
        self.model_normalizer = ModelEvidenceNormalizer()
        self.threshold = None
        
    def fit(self, df_features: pd.DataFrame, df_preds: pd.DataFrame, percentile: float = 99.5):
        """Fit the model normalizer and determine the threshold strictly on dev data."""
        # Fit normalizer
        self.model_normalizer.fit(df_preds)
        
        # Calculate dev scores to find threshold
        df_results = self.predict(df_features, df_preds, is_fit_phase=True)
        scores = df_results['hybrid_anomaly_score'].values
        valid_scores = scores[~np.isnan(scores)]
        self.threshold = np.percentile(valid_scores, percentile)
        
    def _determine_primary_evidence(self, row, all_missing, any_missing):
        """Generates deterministic explanation string."""
        if all_missing:
            return "All sensors missing; probable communication/data-loss event"
        if any_missing:
            return "Sensor missing; communication/data-loss event"
            
        evidence_dict = {
            'Temporal': row['temporal_evidence'],
            'Statistical': row['statistical_evidence'],
            'Multivariate': row['multivariate_evidence'],
            'Stability': row['stability_evidence']
        }
        
        primary = max(evidence_dict, key=evidence_dict.get)
        if evidence_dict[primary] < 0.2:
            if row['suppression_factor'] < 1.0:
                return "High raw deviation but strong multivariate consistency reduced sensor-fault suspicion"
            return "No strong evidence"
            
        if primary == 'Multivariate':
            return "High multivariate disagreement indicating isolated sensor fault"
        elif primary == 'Stability':
            return "Persistent unchanged value increased stability anomaly evidence"
        elif primary == 'Temporal':
            return "Unusual rate of change or temporal deviation"
        else:
            return "Large statistical deviation"
            
    def predict(self, df_features: pd.DataFrame, df_preds: pd.DataFrame, is_fit_phase: bool = False) -> pd.DataFrame:
        if not is_fit_phase and self.threshold is None:
            raise ValueError("Engine must be fitted first.")
            
        # 1. Extract Evidence Layers
        t_ev = extract_temporal_evidence(df_features)
        s_ev = extract_statistical_evidence(df_features, max_z=MAX_Z_CAP)
        m_ev = extract_multivariate_evidence(df_features)
        st_ev = extract_stability_evidence(df_features)
        mis_ev = extract_missing_evidence(df_features)
        mod_ev = self.model_normalizer.transform(df_preds)
        
        # 2. Base Fusion
        base_score = (
            t_ev * FUSION_WEIGHTS['temporal'] +
            s_ev * FUSION_WEIGHTS['statistical'] +
            m_ev * FUSION_WEIGHTS['multivariate'] +
            st_ev * FUSION_WEIGHTS['stability'] +
            mis_ev * FUSION_WEIGHTS['missing'] +
            mod_ev * FUSION_WEIGHTS['model']
        )
        
        # 3. Contextual Suppression Logic
        # Suppress if raw deviation is high (e.g., PCA/Statistical is acting up), 
        # but multivariate disagreement is low (sensors moving coherently) 
        # AND missing is 0 (it's not a missing data event)
        
        # We calculate contextual consistency
        # Low disagreement means high coherence.
        raw_disagreement = df_features['multivariate_z_disagreement'].abs().values
        raw_disagreement = np.nan_to_num(raw_disagreement, nan=0.0)
        
        # suppression_factor = SUPPRESSION_FACTOR if coherent, else 1.0
        suppression_factor = np.ones(len(df_features))
        
        coherent_mask = (raw_disagreement < COHERENT_MULTIVARIATE_THRESHOLD) & (mis_ev == 0)
        suppression_factor[coherent_mask] = SUPPRESSION_FACTOR
        
        # 4. Apply Suppression to get final hybrid score
        hybrid_score = base_score * suppression_factor
        
        # 5. Missing Data Bypass
        # If all missing, force strong anomaly evidence regardless of suppression
        all_missing = df_features['all_sensors_missing'].values
        hybrid_score[all_missing == 1] = np.clip(hybrid_score[all_missing == 1] + 0.8, 0.0, 1.0)
        
        hybrid_score = np.clip(hybrid_score, 0.0, 1.0)
        
        # 6. Confidence Score
        # anomaly_confidence = agreement/consistency among available evidence sources.
        # Calculated as 1.0 - variance of evidence layers (ignoring zeroes if possible, or just standard deviation inverted)
        ev_matrix = np.vstack([t_ev, s_ev, m_ev, st_ev, mis_ev, mod_ev]).T
        # We define confidence heuristically: if multiple independent evidences are high (>0.3), confidence is high.
        # Alternatively, if hybrid score is very high, it requires agreement.
        # We'll use the fraction of active evidence layers (evidence > 0.2).
        active_layers = np.sum(ev_matrix > 0.2, axis=1)
        anomaly_confidence = np.clip(active_layers / 3.0, 0.0, 1.0) # 3+ sources agreeing = 1.0 confidence
        
        # Build Results DataFrame
        df_res = pd.DataFrame()
        df_res['timestamp'] = df_features['timestamp']
        
        df_res['temporal_evidence'] = t_ev
        df_res['statistical_evidence'] = s_ev
        df_res['multivariate_evidence'] = m_ev
        df_res['stability_evidence'] = st_ev
        df_res['missing_evidence'] = mis_ev
        df_res['model_evidence'] = mod_ev
        
        df_res['base_score'] = base_score
        df_res['contextual_consistency'] = (1.0 - m_ev) # High when m_ev is low
        df_res['suppression_factor'] = suppression_factor
        df_res['hybrid_anomaly_score'] = hybrid_score
        df_res['anomaly_confidence'] = anomaly_confidence
        
        if not is_fit_phase:
            df_res['hybrid_prediction'] = (hybrid_score > self.threshold).astype(int)
            
            # Generating primary evidence explanation (vectorized roughly or via apply for simplicity, apply is fine for this benchmark size if optimized)
            # To be fast, we'll use a slightly vectorized approach but a quick apply is easier to implement accurately.
            df_temp = df_res[['temporal_evidence', 'statistical_evidence', 'multivariate_evidence', 'stability_evidence', 'suppression_factor']].copy()
            df_temp['all_missing'] = all_missing
            df_temp['any_missing'] = df_features['any_sensor_missing'].values
            
            # Fast vectorized explanation
            primary_evidence = np.full(len(df_res), "No strong evidence", dtype=object)
            
            ev_cols = ['temporal_evidence', 'statistical_evidence', 'multivariate_evidence', 'stability_evidence']
            ev_names = ['Temporal', 'Statistical', 'Multivariate', 'Stability']
            
            max_ev_idx = np.argmax(df_temp[ev_cols].values, axis=1)
            max_ev_val = np.max(df_temp[ev_cols].values, axis=1)
            
            strong_mask = max_ev_val >= 0.2
            
            # Map index to explanation
            primary_evidence[strong_mask & (max_ev_idx == 2)] = "High multivariate disagreement indicating isolated sensor fault"
            primary_evidence[strong_mask & (max_ev_idx == 3)] = "Persistent unchanged value increased stability anomaly evidence"
            primary_evidence[strong_mask & (max_ev_idx == 0)] = "Unusual rate of change or temporal deviation"
            primary_evidence[strong_mask & (max_ev_idx == 1)] = "Large statistical deviation"
            
            suppressed_mask = (~strong_mask) & (df_temp['suppression_factor'] < 1.0)
            primary_evidence[suppressed_mask] = "High raw deviation but strong multivariate consistency reduced sensor-fault suspicion"
            
            primary_evidence[df_temp['any_missing'] == 1] = "Sensor missing; communication/data-loss event"
            primary_evidence[df_temp['all_missing'] == 1] = "All sensors missing; probable communication/data-loss event"
            
            df_res['primary_evidence'] = primary_evidence
            
        return df_res
