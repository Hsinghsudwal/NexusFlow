from alibi_detect.cd import KSDrift
import numpy as np

class DataDriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.detector = KSDrift(
            x_ref=reference_data,
            p_val=threshold
        )

    def detect_drift(self, current_data):
        preds = self.detector.predict(current_data)
        return {
            'drift_detected': preds['data']['is_drift'],
            'p_value': preds['data']['p_val'],
            'threshold': self.detector.p_val
        }

class ConceptDriftDetector:
    def __init__(self, model, threshold=0.1):
        self.model = model
        self.threshold = threshold

    def detect_drift(self, X, y):
        preds = self.model.predict(X)
        accuracy = np.mean(preds == y)
        return {
            'drift_detected': accuracy < self.threshold,
            'current_accuracy': accuracy
        }