# Hybrid Anomaly Intelligence Configuration

FUSION_WEIGHTS = {
    'temporal': 0.20,
    'statistical': 0.20,
    'multivariate': 0.25,
    'stability': 0.10,
    'missing': 0.15,
    'model': 0.10
}

# The maximum allowed Z-score before capping (prevents 14M scores from 0-variance windows)
MAX_Z_CAP = 100.0

# Contextual Suppression Parameters
# Defines how heavily a score is suppressed if PCA is high but multivariate disagreement is low.
# 1.0 = no suppression. 0.0 = complete suppression.
SUPPRESSION_FACTOR = 0.5 

# Thresholds for context suppression
# If multivariate disagreement is below this, the sensors are acting coherently.
COHERENT_MULTIVARIATE_THRESHOLD = 2.0 
